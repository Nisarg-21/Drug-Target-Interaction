#!/usr/bin/env python
"""Second-pass AlphaFold recovery for the sequences fetch_alphafold.py left without
a usable structure.

Reads alphafold_data/ strictly read-only and writes everything to a separate
directory (default alphafold_recovered/):

    structures/{entryId}_v{N}.cif     recovered AlphaFold mmCIF model
    plddt/{entryId}_v{N}.npy          per-residue pLDDT, float32 [L]
    recovery_map.json                 md5(sequence) -> outcome, candidates tried
    usable_structures_recovered.json  same role as usable_structures.json
    near_misses.json                  >= NEAR_MISS_IDENTITY, logged, never used

Acceptance rule: a structure is used ONLY if the AlphaFold model sequence is
byte-identical to our sequence. Checked twice: the API's uniprotSequence before
download, then the mmCIF's own _entity_poly sequence and the pLDDT length after.

Per fallback reason (from alphafold_data/fallback_list.json):

    unmapped               exact-sequence UniParc lookup (/uniparc/sequence), which
                           returns every cross-reference, inactive and old sequence
                           versions included. The checksum search used by the first
                           pass only exposes uniProtKBAccessions and misses these.
    no_structure           every UniProtKB accession on the same UniParc entry, plus
                           Swiss-Prot isoforms (P12345-2 ...), against the AFDB API.
    model_length_mismatch  as above; also re-checks the model already on disk and
    shared_accession       tries older AFDB model versions if the server offers them.
    residue_mismatch       not in fallback_list.json: first-pass "usable" models with
                           our length but different residues (older UniProt sequence
                           version). Searched like model_length_mismatch.
    skipped_short          skipped.

Recovered records carry match_type: active, inactive (matched accession is an
inactive UniProtKB entry), other_species (organism differs from the first pass's
pinned accession) or shared_group (the first-pass model already on disk, which
exactly matches this member of a shared-accession group).

Polite and resumable like the first pass (shares its PoliteSession): records with
a terminal status in recovery_map.json are not re-queried.

Usage:
    python scripts/recover_alphafold.py --per-category 10 --out-root /tmp/dry   # dry run
    python scripts/recover_alphafold.py                                          # full run
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from difflib import SequenceMatcher

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_alphafold import (  # noqa: E402
    ALPHAFOLD_PREDICTION,
    PoliteSession,
    atomic_write_text,
    extract_plddt,
    load_unique_sequences,
    parse_cif_loop,
)


UNIPARC_SEQUENCE = "https://rest.uniprot.org/uniparc/sequence"
UNIPROTKB_ENTRY = "https://rest.uniprot.org/uniprotkb/{accession}"
AFDB_FILE = "https://alphafold.ebi.ac.uk/files/{entry_id}-model_v{version}.cif"

NEAR_MISS_IDENTITY = 0.95
# Bounds on per-sequence fan-out. Candidates are ordered best-first and the
# search stops at the first exact hit, so these only matter for misses.
MAX_CANDIDATES = 40
MAX_ISOFORM_PARENTS = 3

CATEGORIES = ("unmapped", "no_structure", "model_length_mismatch", "shared_accession",
              "residue_mismatch")
TERMINAL_STATUSES = frozenset({"recovered", "near_miss", "no_match", "no_candidates",
                               "skipped_short"})

SCHEMA_VERSION = 1


# --- sequence helpers ------------------------------------------------------

def identity(ours, theirs):
    """Matched residues / longer length. difflib's matching blocks are a lower
    bound on the true alignment, so this never overstates a near-miss."""
    matcher = SequenceMatcher(None, ours, theirs, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / max(len(ours), len(theirs))


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


def cif_sequence(cif_text):
    """One-letter sequence from the _entity_poly_seq loop (one row per residue).

    Read from the per-residue loop rather than _entity_poly's one-letter field,
    whose layout varies between AFDB releases (key-value vs loop_, with or
    without the _can variant). Unknown monomers become 'X', which never equals a
    dataset residue, so they can only cause a rejection, never a false match.
    """
    columns, rows = parse_cif_loop(cif_text.splitlines(), "_entity_poly_seq.")
    if "mon_id" not in columns:
        raise ValueError("no _entity_poly_seq loop in mmCIF")
    mon_index = columns.index("mon_id")
    return "".join(THREE_TO_ONE.get(row[mon_index], "X")
                   for row in rows if len(row) > mon_index)


# --- candidate collection --------------------------------------------------

def uniparc_xrefs(session, sequence):
    """(upi, [xref dicts]) for UniProtKB cross-references of the exact sequence,
    inactive and isoform entries included. (None, []) when UniParc has no entry."""
    response = session.get(
        UNIPARC_SEQUENCE,
        params={"sequence": sequence},
        expected=(200, 404),
    )
    if response.status_code == 404:
        return None, []
    payload = response.json()
    xrefs = [
        {
            "accession": xref["id"],
            "database": xref["database"],
            "active": bool(xref.get("active")),
            "version": xref.get("version"),
            "organism": (xref.get("organism") or {}).get("scientificName"),
        }
        for xref in payload.get("uniParcCrossReferences", [])
        if xref.get("database", "").startswith("UniProtKB")
    ]
    return payload.get("uniParcId"), xrefs


def isoform_accessions(session, accession):
    """Isoform IDs listed under ALTERNATIVE PRODUCTS for a Swiss-Prot entry."""
    response = session.get(
        UNIPROTKB_ENTRY.format(accession=accession),
        params={"fields": "accession,cc_alternative_products", "format": "json"},
        expected=(200, 400, 404),
    )
    if response.status_code != 200:
        return []
    isoforms = []
    for comment in response.json().get("comments", []):
        if comment.get("commentType") != "ALTERNATIVE PRODUCTS":
            continue
        for isoform in comment.get("isoforms", []):
            for isoform_id in isoform.get("isoformIds", []):
                if isoform_id not in isoforms:
                    isoforms.append(isoform_id)
    return isoforms


def ordered_candidates(xrefs, base_record):
    """Best-first accession list: active Swiss-Prot / isoforms, active TrEMBL,
    the first pass's own candidates, then inactive cross-references."""
    def rank(xref):
        if xref["active"]:
            if "TrEMBL" in xref["database"]:
                return 1
            return 0
        return 3

    ranked = sorted(xrefs, key=lambda x: (rank(x), len(x["accession"]), x["accession"]))
    candidates = OrderedDict()
    for xref in ranked:
        if rank(xref) < 3:
            candidates.setdefault(xref["accession"], "uniparc_active")
    for candidate in base_record.get("candidates", []):
        candidates.setdefault(candidate["accession"], "first_pass_candidate")
    if base_record.get("accession"):
        candidates.setdefault(base_record["accession"], "first_pass_pinned")
    for xref in ranked:
        candidates.setdefault(xref["accession"], "uniparc_inactive")
    return candidates


# --- AlphaFold ---------------------------------------------------------------

class AlphaFoldLookup:
    """Cached AFDB prediction API + older-version file probing."""

    def __init__(self, session):
        self.session = session
        self._entries = {}
        self.old_versions_served = None

    def entries(self, accession):
        if accession not in self._entries:
            response = self.session.get(
                ALPHAFOLD_PREDICTION.format(accession=accession), expected=(200, 400, 404)
            )
            self._entries[accession] = response.json() if response.status_code == 200 else []
        return self._entries[accession]

    def probe_old_versions(self, entry_id="AF-Q9HBW0-F1", version=4):
        """One request at startup: does the AFDB file server still hand out
        pre-latest model versions? If not, skip them for every sequence."""
        response = self.session.get(
            AFDB_FILE.format(entry_id=entry_id, version=version), expected=(200, 403, 404)
        )
        self.old_versions_served = response.status_code == 200
        return response.status_code

    def old_version_text(self, entry_id, version):
        if not self.old_versions_served:
            return None
        response = self.session.get(
            AFDB_FILE.format(entry_id=entry_id, version=version), expected=(200, 403, 404)
        )
        return response.text if response.status_code == 200 else None


def repo_relative(path, repo_root):
    """Repo-relative when inside the repo (portable), absolute otherwise."""
    path = os.path.abspath(path)
    return os.path.relpath(path, repo_root) if path.startswith(repo_root + os.sep) else path


def save_structure(cif_text, sequence, file_stem, structures_dir, plddt_dir, repo_root):
    """Verify against the file itself, then write cif + pLDDT under new names.
    Existing files are never overwritten (a resumed run reuses identical ones)."""
    if cif_sequence(cif_text) != sequence:
        raise ValueError(f"{file_stem}: mmCIF sequence differs from API sequence")
    plddt = extract_plddt(cif_text)
    if plddt.shape[0] != len(sequence):
        raise ValueError(f"{file_stem}: pLDDT length {plddt.shape[0]} != {len(sequence)}")

    cif_path = os.path.join(structures_dir, f"{file_stem}.cif")
    plddt_path = os.path.join(plddt_dir, f"{file_stem}.npy")
    if not os.path.exists(cif_path):
        atomic_write_text(cif_path, cif_text)
    if not os.path.exists(plddt_path):
        temp_path = f"{plddt_path}.tmp.{os.getpid()}.npy"
        np.save(temp_path, plddt)
        os.replace(temp_path, plddt_path)
    return {
        "structure_path": repo_relative(cif_path, repo_root),
        "plddt_path": repo_relative(plddt_path, repo_root),
        "plddt_length": int(plddt.shape[0]),
        "plddt_mean": round(float(plddt.mean()), 2),
    }


# --- match provenance ----------------------------------------------------------

def reference_organism(lookup, xrefs, base_record):
    """The organism this sequence is assumed to come from: that of the first
    pass's pinned accession, else of a Swiss-Prot cross-reference (active first),
    else of any cross-reference. None when nothing says."""
    pinned = base_record.get("accession")
    if pinned:
        for xref in xrefs:
            if xref["accession"] == pinned and xref["organism"]:
                return xref["organism"]
        for entry in lookup.entries(pinned):
            if entry.get("organismScientificName"):
                return entry["organismScientificName"]
    ranked = sorted(xrefs, key=lambda x: ("Swiss-Prot" not in x["database"], not x["active"]))
    for xref in ranked:
        if xref["organism"]:
            return xref["organism"]
    return None


def classify_match(lookup, accession, entry, xrefs, base_record):
    """match_type for filtering: inactive > other_species > active. All three
    are exact sequence matches; match_flags keeps every flag that applies."""
    flags = []
    xref = next((x for x in xrefs if x["accession"] == accession), None)
    if xref is not None and not xref["active"]:
        flags.append("inactive")
    reference = reference_organism(lookup, xrefs, base_record)
    matched = entry.get("organismScientificName")
    if reference and matched and reference != matched:
        flags.append("other_species")
    return {
        "match_type": flags[0] if flags else "active",
        "match_flags": flags,
        "reference_organism": reference,
        "matched_organism": matched,
    }


# --- per-sequence driver -------------------------------------------------------

def recover_one(session, lookup, sequence, category, base_record, af_root, dirs, repo_root):
    record = {
        "category": category,
        "sequence_length": len(sequence),
        "uniparc_id": None,
        "tried": [],
        "best": None,
        "note": None,
    }
    if category == "skipped_short":
        record["status"] = "skipped_short"
        return record

    # Shared-accession records: the model already on disk may be an exact match
    # for this particular sequence (it matches at most one of the sharers).
    existing = base_record.get("structure_file")
    if existing:
        existing_path = os.path.join(af_root, existing)
        with open(existing_path, "r", encoding="utf-8") as handle:
            existing_seq = cif_sequence(handle.read())
        if existing_seq == sequence and len(sequence) == base_record.get("plddt_length"):
            record.update({
                "status": "recovered",
                "source": "existing_first_pass_file",
                "accession": base_record["accession"],
                "model_entry_id": base_record.get("model_entry_id"),
                "structure_path": repo_relative(existing_path, repo_root),
                "plddt_path": repo_relative(
                    os.path.join(af_root, base_record["plddt_file"]), repo_root),
                "plddt_length": base_record["plddt_length"],
                "match_type": "shared_group",
                "match_flags": ["shared_group"],
            })
            return record
        record["tried"].append({"accession": base_record["accession"], "source": "existing_file",
                                "identity": round(identity(sequence, existing_seq), 4),
                                "model_length": len(existing_seq)})

    upi, xrefs = uniparc_xrefs(session, sequence)
    record["uniparc_id"] = upi
    record["uniparc_uniprot_xrefs"] = len(xrefs)
    candidates = ordered_candidates(xrefs, base_record)

    # Isoforms of the leading Swiss-Prot parents (base accession, no suffix).
    parents = [acc for acc, src in candidates.items()
               if "-" not in acc and src in ("uniparc_active", "first_pass_pinned",
                                             "first_pass_candidate")
               and len(acc) == 6][:MAX_ISOFORM_PARENTS]
    for parent in parents:
        for isoform in isoform_accessions(session, parent):
            candidates.setdefault(isoform, "isoform")

    if not candidates:
        record["status"] = "no_candidates"
        record["note"] = ("sequence not in UniParc" if upi is None
                          else "UniParc entry has no UniProtKB cross-references")
        return record

    best = None
    for accession, source in list(candidates.items())[:MAX_CANDIDATES]:
        entries = lookup.entries(accession)
        if not entries:
            record["tried"].append({"accession": accession, "source": source, "afdb": False})
            continue
        for entry in entries:
            model_seq = entry.get("uniprotSequence") or entry.get("sequence") or ""
            version = entry.get("latestVersion")
            entry_id = entry.get("entryId")
            if model_seq == sequence:
                response = session.get(entry["cifUrl"])
                if response.status_code != 200:
                    raise RuntimeError(f"cif download returned HTTP {response.status_code}")
                saved = save_structure(response.text, sequence, f"{entry_id}_v{version}",
                                       dirs["structures"], dirs["plddt"], repo_root)
                record.update(saved)
                record.update({"status": "recovered", "source": source,
                               "accession": accession, "model_entry_id": entry_id,
                               "model_version": version})
                record.update(classify_match(lookup, accession, entry, xrefs, base_record))
                return record

            score = identity(sequence, model_seq)
            attempt = {"accession": accession, "source": source, "afdb": True,
                       "entry_id": entry_id, "version": version,
                       "model_length": len(model_seq), "identity": round(score, 4),
                       "ours_is_substring": sequence in model_seq}

            # Older model versions may have been built from our (older) sequence.
            for old in sorted(set(entry.get("allVersions", [])) - {version}, reverse=True):
                old_text = lookup.old_version_text(entry_id, old)
                if old_text is None:
                    continue
                if cif_sequence(old_text) == sequence:
                    saved = save_structure(old_text, sequence, f"{entry_id}_v{old}",
                                           dirs["structures"], dirs["plddt"], repo_root)
                    record.update(saved)
                    record.update({"status": "recovered", "source": f"{source}+old_version",
                                   "accession": accession, "model_entry_id": entry_id,
                                   "model_version": old})
                    record.update(classify_match(lookup, accession, entry, xrefs, base_record))
                    return record

            record["tried"].append(attempt)
            if best is None or score > best["identity"]:
                best = attempt

    record["best"] = best
    if best is not None and best["identity"] >= NEAR_MISS_IDENTITY:
        record["status"] = "near_miss"
    else:
        record["status"] = "no_match"
        if best is None:
            record["note"] = "no candidate accession has an AlphaFold model"
    return record


# --- outputs -------------------------------------------------------------------

def write_outputs(records, out_root):
    def dump(name, meaning, payload):
        atomic_write_text(os.path.join(out_root, name), json.dumps({
            "schema_version": SCHEMA_VERSION,
            "generated_by": "scripts/recover_alphafold.py",
            "key": "md5(sequence)",
            "meaning": meaning,
            "count": len(payload),
            "records": payload,
        }, indent=2, sort_keys=True))

    dump("recovery_map.json", "per-sequence recovery outcome for fallback_list.json entries; "
         "recovered records carry match_type (active / inactive / other_species / shared_group)",
         records)
    dump("usable_structures_recovered.json",
         "exact sequence<->model matches; paths are relative to the repo root",
         {key: {field: r[field] for field in ("accession", "model_entry_id", "structure_path",
                                              "plddt_path", "plddt_length", "source",
                                              "match_type")}
          | {"sequence_length": r["sequence_length"]}
          for key, r in records.items() if r["status"] == "recovered"})
    dump("near_misses.json", f"best AF model identity >= {NEAR_MISS_IDENTITY}; NOT used",
         {key: {"category": r["category"], "sequence_length": r["sequence_length"],
                "best": r["best"]}
          for key, r in records.items() if r["status"] == "near_miss"})


def summarize(records, keys):
    table = {}
    for key in keys:
        record = records.get(key)
        if record is None:
            continue
        row = table.setdefault(record["category"], {})
        row[record["status"]] = row.get(record["status"], 0) + 1
    for category in CATEGORIES + ("skipped_short",):
        if category in table:
            print(f"  {category:<22} " + "  ".join(f"{s}={n}" for s, n in sorted(table[category].items())))


def main():
    parser = argparse.ArgumentParser(description="Recover AlphaFold structures for fallback sequences.")
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--datasets", nargs="+", default=["biosnap", "bindingdb"])
    parser.add_argument("--af-root", default="alphafold_data",
                        help="first-pass output (read-only)")
    parser.add_argument("--out-root", default="alphafold_recovered",
                        help="where recovered files and maps are written")
    parser.add_argument("--per-category", type=int, default=None,
                        help="process at most N pending sequences per category (dry run)")
    parser.add_argument("--delay", type=float, default=0.34)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    repo_root = os.getcwd()
    af_root = os.path.abspath(args.af_root)
    out_root = os.path.abspath(args.out_root)
    if out_root == af_root or out_root.startswith(af_root + os.sep):
        parser.error("--out-root must be outside --af-root; the first-pass data is read-only")
    dirs = {"structures": os.path.join(out_root, "structures"),
            "plddt": os.path.join(out_root, "plddt")}
    for directory in (out_root, *dirs.values()):
        os.makedirs(directory, exist_ok=True)

    with open(os.path.join(af_root, "fallback_list.json"), "r", encoding="utf-8") as handle:
        fallback = json.load(handle)["records"]
    with open(os.path.join(af_root, "uniprot_map.json"), "r", encoding="utf-8") as handle:
        base_records = json.load(handle)["records"]
    sequences = load_unique_sequences(args.datasets_root, args.datasets)

    # Targets: every fallback sequence, plus first-pass "usable" records whose model
    # has our length but not our residues (an older UniProt sequence version of the
    # same length - invisible to the first pass's length check).
    targets = {key: entry["reason"] for key, entry in fallback.items()}
    with open(os.path.join(af_root, "usable_structures.json"), "r", encoding="utf-8") as handle:
        usable = json.load(handle)["records"]
    for key, entry in usable.items():
        with open(os.path.join(af_root, entry["structure_file"]), "r", encoding="utf-8") as handle:
            if cif_sequence(handle.read()) != sequences[key]["sequence"]:
                targets[key] = "residue_mismatch"
    residue_mismatch = sum(1 for c in targets.values() if c == "residue_mismatch")
    print(f"first-pass usable with residue mismatch: {residue_mismatch}")

    map_path = os.path.join(out_root, "recovery_map.json")
    records = {}
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)["records"]
        print(f"resuming: {len(records)} records in {map_path}")

    pending, per_category = [], {}
    for key in sorted(targets):
        if records.get(key, {}).get("status") in TERMINAL_STATUSES:
            continue
        category = targets[key]
        if args.per_category is not None and per_category.get(category, 0) >= args.per_category:
            continue
        per_category[category] = per_category.get(category, 0) + 1
        pending.append(key)
    print(f"targets: {len(targets)}   to process this run: {len(pending)} {per_category}")

    session = PoliteSession(delay=args.delay, retries=args.retries, timeout=args.timeout)
    lookup = AlphaFoldLookup(session)
    code = lookup.probe_old_versions()
    print(f"older AFDB model versions served: {lookup.old_versions_served} (probe HTTP {code})")

    interrupted = False
    try:
        for position, key in enumerate(pending, start=1):
            category = targets[key]
            try:
                records[key] = recover_one(session, lookup, sequences[key]["sequence"], category,
                                           base_records.get(key, {}), af_root, dirs, repo_root)
            except (RuntimeError, ValueError, KeyError, requests.RequestException, OSError) as exc:
                records[key] = {"category": category, "status": "error",
                                "sequence_length": len(sequences[key]["sequence"]),
                                "note": f"{type(exc).__name__}: {exc}"}
            print(f"  [{position}/{len(pending)}] {key[:10]} {category:<22} "
                  f"-> {records[key]['status']}"
                  + (f" ({records[key].get('accession')}, {records[key].get('source')})"
                     if records[key]["status"] == "recovered" else "")
                  + (f" (best {records[key]['best']['accession']} "
                     f"id={records[key]['best']['identity']})"
                     if records[key].get("best") else ""), flush=True)
            if position % args.checkpoint_every == 0:
                write_outputs(records, out_root)
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted - saving progress", flush=True)

    write_outputs(records, out_root)
    print("\nthis run:")
    summarize(records, pending)
    print("all records so far:")
    summarize(records, list(records))
    match_types = {}
    for r in records.values():
        if r["status"] == "recovered":
            match_types[r["match_type"]] = match_types.get(r["match_type"], 0) + 1
    print(f"recovered by match_type: {match_types}")
    recovered = sum(1 for r in records.values() if r["status"] == "recovered")
    usable_first = len(sequences) - len(targets)
    print(f"\nfirst-pass exact usable {usable_first} + recovered {recovered} "
          f"= {usable_first + recovered} / {len(sequences)}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
