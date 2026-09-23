#!/usr/bin/env python
"""One-time AlphaFold structure download + UniProt mapping job.

Reads the unique protein sequences out of the dataset CSVs, resolves each one to
a single UniProtKB accession via UniParc's CRC64 checksum index, and pulls the
matching AlphaFold model. CPU and network only: nothing here loads a model,
trains, or touches the training code.

Everything is written under a single portable directory (default alphafold_data/):

    structures/{ACC}.cif    AlphaFold mmCIF model
    plddt/{ACC}.npy         per-residue pLDDT, float32 [L]
    uniprot_map.json        md5(sequence) -> accession, pin rule, status
    coverage_report.txt     miss / edge-case report

The datasets are read-only and no training file is touched.

Idempotent and resumable: uniprot_map.json is the job's state. Sequences that
reached a terminal status are skipped on a re-run, already-downloaded files are
not re-fetched, and Ctrl-C saves progress before exiting.

Why CRC64 and not a name lookup: UniParc indexes by checksum, so a hit means the
stored sequence is byte-identical to ours. That is the only mapping that is safe
to key an embedding cache on. Verified against the API's own crc64 field.

Usage:
    python scripts/fetch_alphafold.py --limit 25     # trial
    python scripts/fetch_alphafold.py                # full run
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import requests


# --- endpoints -------------------------------------------------------------

UNIPARC_SEARCH = "https://rest.uniprot.org/uniparc/search"
UNIPROTKB_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD_PREDICTION = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"

USER_AGENT = "CMA-DTI-alphafold-fetch/1.0 (academic research; one-time bulk fetch)"

# --- thresholds ------------------------------------------------------------

# AlphaFold DB does not model sequences beyond this length.
ALPHAFOLD_MAX_LENGTH = 2700
# Short peptides are usually not independent UniProtKB entries.
MIN_PEPTIDE_LENGTH = 50
# Selenocysteine and the unknown-residue placeholder. Flagged, never rewritten:
# substituting them would change the sequence and therefore the md5 cache key.
NONSTANDARD_RESIDUES = ("U", "X")

# Statuses that will not be retried on a re-run.
TERMINAL_STATUSES = frozenset({
    "ok",               # structure + pLDDT on disk
    "no_structure",     # accession pinned, AlphaFold returned 404 (a real miss)
    "af_bad_request",   # AlphaFold returned 400 (malformed ID: an error, not a miss)
    "unmapped",         # UniParc gave no usable active accession
    "skipped_short",    # below MIN_PEPTIDE_LENGTH, never queried
})

SCHEMA_VERSION = 1


# --- checksums -------------------------------------------------------------

# CRC-64-ISO as used by UniParc (the SWISS-PROT crc64 with POLY64REV).
_POLY64REV = 0xD800000000000000


def _build_crc64_table():
    table = []
    for index in range(256):
        part = index
        for _ in range(8):
            part = (part >> 1) ^ _POLY64REV if part & 1 else part >> 1
        table.append(part)
    return table


_CRC64_TABLE = _build_crc64_table()


def crc64(sequence):
    """UniParc-compatible CRC64, uppercase hex. Matches the API's crc64 field."""
    crc = 0
    for byte in sequence.encode("ascii"):
        crc = _CRC64_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return "%016X" % crc


def sequence_md5(sequence):
    """Same keying as the encoder cache (models.py:70-71, protein_3d.py)."""
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


# --- HTTP ------------------------------------------------------------------

class PoliteSession:
    """Rate-limited requests wrapper with backoff on transient failures.

    Expected non-200 codes (404 for a missing AlphaFold model, 400 for a bad
    accession) are returned to the caller rather than retried: they are answers,
    not failures. 429 and 5xx are retried with exponential backoff, honouring
    Retry-After when the server sends one.
    """

    def __init__(self, delay, retries, timeout):
        self.delay = delay
        self.retries = retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_request = 0.0

    def _throttle(self):
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.monotonic()

    def get(self, url, params=None, expected=(200,)):
        last_error = None
        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code in expected:
                    return response
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        time.sleep(min(int(retry_after), 60))
                        continue
                else:
                    # An unexpected 4xx is a definite answer; hand it back.
                    return response

            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 30))

        raise RuntimeError(f"{url} failed after {self.retries + 1} attempts ({last_error})")


# --- input -----------------------------------------------------------------

def load_unique_sequences(datasets_root, dataset_names):
    """Unique protein sequences across the datasets, keyed by md5.

    Reads full.csv when present, otherwise concatenates whatever split CSVs the
    dataset directory holds. Read-only: nothing under datasets/ is written.
    """
    records = OrderedDict()

    for name in dataset_names:
        dataset_dir = os.path.join(datasets_root, name)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")

        full_csv = os.path.join(dataset_dir, "full.csv")
        if os.path.exists(full_csv):
            csv_paths = [full_csv]
        else:
            csv_paths = sorted(
                os.path.join(root, filename)
                for root, _, filenames in os.walk(dataset_dir)
                for filename in filenames
                if filename.endswith(".csv")
            )
            if not csv_paths:
                raise FileNotFoundError(f"no CSVs found under {dataset_dir}")

        for csv_path in csv_paths:
            frame = pd.read_csv(csv_path, usecols=["Protein"])
            for sequence in frame["Protein"].astype(str):
                key = sequence_md5(sequence)
                record = records.get(key)
                if record is None:
                    records[key] = {"sequence": sequence, "datasets": [name]}
                elif name not in record["datasets"]:
                    record["datasets"].append(name)

    return records


def sequence_flags(sequence):
    """Edge-case markers recorded for every sequence, hit or miss."""
    flags = []
    if len(sequence) > ALPHAFOLD_MAX_LENGTH:
        flags.append("over_alphafold_max_length")
    if len(sequence) < MIN_PEPTIDE_LENGTH:
        flags.append("short_peptide")
    for residue in NONSTANDARD_RESIDUES:
        if residue in sequence:
            flags.append(f"nonstandard_{residue}")
    return flags


# --- mapping ---------------------------------------------------------------

def uniparc_accessions(session, checksum):
    """Active UniProtKB accessions whose sequence is byte-identical to ours.

    The search endpoint returns uniProtKBAccessions directly, mixing versioned
    ("O95363.1") and bare ("O95363") forms, so versions are stripped and the
    list deduplicated with order preserved.
    """
    response = session.get(
        UNIPARC_SEARCH,
        params={"query": f"checksum:{checksum}", "format": "json", "size": 1},
    )
    if response.status_code != 200:
        raise RuntimeError(f"UniParc search returned HTTP {response.status_code}")

    results = response.json().get("results", [])
    if not results:
        return []

    seen = OrderedDict()
    for raw in results[0].get("uniProtKBAccessions", []):
        seen.setdefault(raw.split(".", 1)[0], None)
    return list(seen)


def classify_accessions(session, accessions, batch_size=20):
    """Map accession -> entryType ('... reviewed (Swiss-Prot)', '... unreviewed
    (TrEMBL)', 'Inactive'). Accessions the search does not return are unknown."""
    classified = {}
    for start in range(0, len(accessions), batch_size):
        batch = accessions[start:start + batch_size]
        query = " OR ".join(f"accession:{accession}" for accession in batch)
        response = session.get(
            UNIPROTKB_SEARCH,
            params={"query": query, "fields": "accession,reviewed", "format": "json"},
        )
        if response.status_code != 200:
            raise RuntimeError(f"UniProtKB search returned HTTP {response.status_code}")
        for entry in response.json().get("results", []):
            accession = entry.get("primaryAccession")
            if accession:
                classified[accession] = entry.get("entryType", "unknown")
    return classified


def pin_accession(classified):
    """Pick exactly one accession, deterministically, and say which rule fired.

    Order: reviewed (Swiss-Prot) first, then shortest accession string, then
    alphabetical. Inactive entries are dropped entirely - they carry no
    AlphaFold model and would turn into fake 404 misses.

    Stability matters more than choosing the "best" ortholog: the accession
    becomes part of the on-disk cache identity, so the same sequence must pin
    the same accession on every re-run.
    """
    candidates = [
        (accession, entry_type)
        for accession, entry_type in classified.items()
        if entry_type != "Inactive"
    ]
    if not candidates:
        return None, None, []

    def sort_key(item):
        accession, entry_type = item
        return (0 if "reviewed (Swiss-Prot)" in entry_type else 1, len(accession), accession)

    ordered = sorted(candidates, key=sort_key)
    chosen, chosen_type = ordered[0]

    if len(ordered) == 1:
        rule = "sole_candidate"
    else:
        chosen_tier = sort_key(ordered[0])[0]
        same_tier = [item for item in ordered if sort_key(item)[0] == chosen_tier]
        if len(same_tier) < len(ordered):
            rule = "reviewed_preferred"
        elif len([i for i in same_tier if len(i[0]) == len(chosen)]) == 1:
            rule = "shortest_accession"
        else:
            rule = "alphabetical"

    audit = [{"accession": a, "entry_type": t} for a, t in ordered]
    return chosen, rule, audit


# --- AlphaFold -------------------------------------------------------------

def alphafold_entry(session, accession):
    """(status_code, entry dict or None). 404 = genuine miss, 400 = bad ID."""
    response = session.get(
        ALPHAFOLD_PREDICTION.format(accession=accession), expected=(200, 404, 400)
    )
    if response.status_code != 200:
        return response.status_code, None
    payload = response.json()
    return 200, (payload[0] if payload else None)


def parse_cif_loop(lines, prefix):
    """Return (column names, rows) for the mmCIF loop_ whose headers start with
    prefix. Minimal on purpose - AlphaFold files are machine-generated and the
    two categories read here hold no quoted multi-token values."""
    total = len(lines)
    index = 0
    while index < total:
        if lines[index].strip() == "loop_" and index + 1 < total and lines[index + 1].startswith(prefix):
            columns = []
            cursor = index + 1
            while cursor < total and lines[cursor].startswith(prefix):
                columns.append(lines[cursor].strip().split(".", 1)[1])
                cursor += 1
            rows = []
            while cursor < total:
                stripped = lines[cursor].strip()
                if not stripped or stripped in ("#", "loop_") or stripped.startswith("_"):
                    break
                rows.append(lines[cursor].split())
                cursor += 1
            return columns, rows
        index += 1
    return [], []


def extract_plddt(cif_text):
    """Per-residue pLDDT as float32 [L].

    Primary source is _ma_qa_metric_local, AlphaFold's own per-residue
    confidence category. Falls back to CA B-factors, which the format sets to
    the same values (verified identical on a real model).
    """
    lines = cif_text.splitlines()

    columns, rows = parse_cif_loop(lines, "_ma_qa_metric_local.")
    if columns and "metric_value" in columns:
        value_index = columns.index("metric_value")
        values = [float(row[value_index]) for row in rows if len(row) > value_index]
        if values:
            return np.asarray(values, dtype=np.float32)

    columns, rows = parse_cif_loop(lines, "_atom_site.")
    if columns and "B_iso_or_equiv" in columns and "label_atom_id" in columns:
        b_index = columns.index("B_iso_or_equiv")
        atom_index = columns.index("label_atom_id")
        values = [
            float(row[b_index])
            for row in rows
            if len(row) > max(b_index, atom_index) and row[atom_index] == "CA"
        ]
        if values:
            return np.asarray(values, dtype=np.float32)

    raise ValueError("no pLDDT found in mmCIF (neither _ma_qa_metric_local nor CA B-factors)")


def atomic_write_bytes(path, payload):
    """Temp file + replace, so an interrupted write cannot leave a truncated
    file that a resumed run would treat as already downloaded."""
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(temp_path, "wb") as handle:
        handle.write(payload)
    os.replace(temp_path, path)


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def download_structure_and_plddt(session, entry, accession, structures_dir, plddt_dir, force):
    """Fetch the mmCIF and derive pLDDT. Returns a dict of result fields."""
    cif_path = os.path.join(structures_dir, f"{accession}.cif")
    plddt_path = os.path.join(plddt_dir, f"{accession}.npy")

    have_cif = os.path.exists(cif_path) and os.path.getsize(cif_path) > 0
    have_plddt = os.path.exists(plddt_path) and os.path.getsize(plddt_path) > 0

    if have_cif and have_plddt and not force:
        plddt = np.load(plddt_path)
    else:
        if have_cif and not force:
            with open(cif_path, "r", encoding="utf-8") as handle:
                cif_text = handle.read()
        else:
            cif_url = entry.get("cifUrl")
            if not cif_url:
                raise RuntimeError(f"AlphaFold entry for {accession} has no cifUrl")
            response = session.get(cif_url)
            if response.status_code != 200:
                raise RuntimeError(f"cif download returned HTTP {response.status_code}")
            cif_text = response.text
            atomic_write_text(cif_path, cif_text)

        plddt = extract_plddt(cif_text)
        # np.save appends .npy unless the path already ends in it, so the temp
        # name carries the suffix and the replace target is unambiguous.
        temp_path = f"{plddt_path}.tmp.{os.getpid()}.npy"
        np.save(temp_path, plddt)
        os.replace(temp_path, plddt_path)

    return {
        "structure_file": os.path.relpath(cif_path, os.path.dirname(structures_dir)).replace("\\", "/"),
        "plddt_file": os.path.relpath(plddt_path, os.path.dirname(plddt_dir)).replace("\\", "/"),
        "plddt_length": int(plddt.shape[0]),
        "plddt_mean": round(float(plddt.mean()), 2),
        "model_entry_id": entry.get("entryId"),
    }


# --- state -----------------------------------------------------------------

def load_map(map_path):
    if not os.path.exists(map_path):
        return {}
    with open(map_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("records", {})


def save_map(map_path, records):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/fetch_alphafold.py",
        "key": "md5(sequence)",
        "records": records,
    }
    atomic_write_text(map_path, json.dumps(payload, indent=2, sort_keys=True))


# --- derived selection lists -----------------------------------------------

def annotate_shared_accessions(records):
    """Flag every downloaded record whose accession is pinned by another sequence.

    One UniProtKB accession cross-references several UniParc sequence versions,
    so two sequences with different md5s - and different CRC64s - can each
    legitimately pin the same accession and end up pointing at the same model.
    That model matches at most one of them.

    model_length_mismatch only catches this when the lengths happen to differ.
    P61168 is pinned by two distinct 444-residue sequences and slips straight
    through it, so the sharing itself has to be flagged separately.

    Recomputed from scratch on every call, so it stays correct on a resumed run
    where later sequences introduce a collision with earlier ones.
    """
    counts = {}
    for record in records.values():
        if record.get("status") == "ok" and record.get("accession"):
            counts[record["accession"]] = counts.get(record["accession"], 0) + 1

    flagged = 0
    for record in records.values():
        flags = [flag for flag in record.get("flags", []) if flag != "shared_accession"]
        if record.get("status") == "ok" and counts.get(record.get("accession"), 0) > 1:
            flags.append("shared_accession")
            flagged += 1
        record["flags"] = flags
    return flagged


def fallback_reason(record):
    """Why this sequence cannot use a structure, or None if it can.

    A record is fallback-only if it has no structure at all, or if it has one
    that cannot be trusted to line up residue-for-residue with our sequence.
    """
    status = record["status"]
    if status in ("unmapped", "no_structure", "skipped_short", "af_bad_request"):
        return status
    if status != "ok":
        return None                       # transient error: undecided, neither list
    if "model_length_mismatch" in record["flags"]:
        return "model_length_mismatch"
    if "shared_accession" in record["flags"]:
        return "shared_accession"
    return None


def write_selection_lists(records, out_root):
    """Split the map into the two lists the precompute step consumes.

        fallback_list.json      must use the ESM-2 fallback (has_structure=0)
        usable_structures.json  clean 1:1 sequence<->model, safe for ESM-IF1

    Every terminal record lands in exactly one list. Records still in 'error'
    land in neither and are returned separately: they are unresolved, not
    decided, and a re-run may still turn them into either.
    """
    fallback = {}
    usable = {}
    unresolved = []

    for key in sorted(records):
        record = records[key]
        if record["status"] == "error":
            unresolved.append(key)
            continue

        reason = fallback_reason(record)
        if reason is None:
            usable[key] = {
                "accession": record["accession"],
                "structure_file": record["structure_file"],
                "plddt_file": record["plddt_file"],
                "sequence_length": record["sequence_length"],
                "plddt_length": record["plddt_length"],
            }
        else:
            fallback[key] = {
                "reason": reason,
                "sequence_length": record["sequence_length"],
                "accession": record.get("accession"),
            }

    reason_counts = {}
    for entry in fallback.values():
        reason_counts[entry["reason"]] = reason_counts.get(entry["reason"], 0) + 1

    atomic_write_text(
        os.path.join(out_root, "fallback_list.json"),
        json.dumps({
            "schema_version": SCHEMA_VERSION,
            "generated_by": "scripts/fetch_alphafold.py",
            "key": "md5(sequence)",
            "meaning": "encode these with has_structure=0 (ESM-2 fallback, 1280-wide)",
            "count": len(fallback),
            "reason_counts": reason_counts,
            "records": fallback,
        }, indent=2, sort_keys=True),
    )
    atomic_write_text(
        os.path.join(out_root, "usable_structures.json"),
        json.dumps({
            "schema_version": SCHEMA_VERSION,
            "generated_by": "scripts/fetch_alphafold.py",
            "key": "md5(sequence)",
            "meaning": "clean 1:1 sequence<->model; run ESM-IF1 on these (has_structure=1)",
            "count": len(usable),
            "records": usable,
        }, indent=2, sort_keys=True),
    )

    return len(fallback), len(usable), reason_counts, unresolved


# --- report ----------------------------------------------------------------

def build_report(records, dataset_names):
    total = len(records)
    counts = {}
    for record in records.values():
        counts[record["status"]] = counts.get(record["status"], 0) + 1

    mapped = sum(1 for r in records.values() if r.get("accession"))
    with_structure = counts.get("ok", 0)
    misses = counts.get("no_structure", 0)

    lines = []
    lines.append("AlphaFold coverage report")
    lines.append("=" * 72)
    lines.append(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("COMBINED")
    lines.append(f"  unique sequences          : {total}")
    lines.append(f"  mapped to an accession    : {mapped}"
                 + (f"  ({mapped / total:.1%})" if total else ""))
    lines.append(f"  structure downloaded      : {with_structure}"
                 + (f"  ({with_structure / total:.1%})" if total else ""))
    lines.append(f"  genuine 404 misses        : {misses}")
    lines.append("")
    lines.append("  status breakdown:")
    for status in sorted(counts):
        lines.append(f"    {status:<18} {counts[status]}")
    lines.append("")

    lines.append("PER DATASET")
    for name in dataset_names:
        subset = [r for r in records.values() if name in r["datasets"]]
        sub_total = len(subset)
        sub_mapped = sum(1 for r in subset if r.get("accession"))
        sub_struct = sum(1 for r in subset if r["status"] == "ok")
        lines.append(f"  {name}")
        lines.append(f"    unique sequences        : {sub_total}")
        lines.append(f"    mapped to an accession  : {sub_mapped}"
                     + (f"  ({sub_mapped / sub_total:.1%})" if sub_total else ""))
        lines.append(f"    structure downloaded    : {sub_struct}"
                     + (f"  ({sub_struct / sub_total:.1%})" if sub_total else ""))
    lines.append("")

    def section(title, predicate, formatter):
        selected = [(k, r) for k, r in sorted(records.items()) if predicate(r)]
        lines.append(f"{title} ({len(selected)})")
        lines.append("-" * 72)
        if not selected:
            lines.append("  (none)")
        for key, record in selected:
            lines.append("  " + formatter(key, record))
        lines.append("")

    section(
        "UNMAPPED - no active UniProtKB accession for the checksum",
        lambda r: r["status"] == "unmapped",
        lambda k, r: f"{k}  len={r['sequence_length']}  crc64={r['crc64']}",
    )
    section(
        "404 MISSES - accession valid, no AlphaFold model",
        lambda r: r["status"] == "no_structure",
        lambda k, r: f"{k}  {r['accession']}  len={r['sequence_length']}",
    )
    section(
        "ERRORS - AlphaFold HTTP 400 (malformed ID; not counted as a miss)",
        lambda r: r["status"] == "af_bad_request",
        lambda k, r: f"{k}  {r.get('accession')}  len={r['sequence_length']}",
    )
    section(
        "UNRESOLVED - transient failures, retried on the next run",
        lambda r: r["status"] == "error",
        lambda k, r: f"{k}  len={r['sequence_length']}  {r.get('note')}",
    )
    section(
        f"OVER {ALPHAFOLD_MAX_LENGTH} aa - beyond the AlphaFold length cap",
        lambda r: "over_alphafold_max_length" in r["flags"],
        lambda k, r: f"{k}  len={r['sequence_length']}  status={r['status']}  acc={r.get('accession')}",
    )
    section(
        f"UNDER {MIN_PEPTIDE_LENGTH} aa - short peptides, skipped without querying",
        lambda r: "short_peptide" in r["flags"],
        lambda k, r: f"{k}  len={r['sequence_length']}  status={r['status']}",
    )
    section(
        "NON-STANDARD RESIDUES - contains U (selenocysteine) or X (unknown)",
        lambda r: any(f.startswith("nonstandard_") for f in r["flags"]),
        lambda k, r: (f"{k}  len={r['sequence_length']}  "
                      f"{','.join(f for f in r['flags'] if f.startswith('nonstandard_'))}  "
                      f"status={r['status']}"),
    )
    section(
        "LENGTH MISMATCH - model length differs from our sequence length",
        lambda r: "model_length_mismatch" in r["flags"],
        lambda k, r: (f"{k}  {r.get('accession')}  seq={r['sequence_length']}  "
                      f"model={r.get('plddt_length')}"),
    )
    section(
        "SHARED ACCESSION - one model pinned by more than one distinct sequence",
        lambda r: "shared_accession" in r["flags"],
        lambda k, r: (f"{k}  {r.get('accession')}  seq={r['sequence_length']}  "
                      f"model={r.get('plddt_length')}"),
    )

    return "\n".join(lines)


# --- driver ----------------------------------------------------------------

def process_sequence(session, key, entry, records, structures_dir, plddt_dir, force):
    sequence = entry["sequence"]
    record = {
        "sequence_length": len(sequence),
        "crc64": crc64(sequence),
        "datasets": entry["datasets"],
        "flags": sequence_flags(sequence),
        "accession": None,
        "pin_rule": None,
        "candidates": [],
        "note": None,
    }

    # Short peptides: logged and skipped, no network call.
    if len(sequence) < MIN_PEPTIDE_LENGTH:
        record["status"] = "skipped_short"
        record["note"] = f"below {MIN_PEPTIDE_LENGTH} aa; not queried"
        records[key] = record
        return

    try:
        accessions = uniparc_accessions(session, record["crc64"])
        if not accessions:
            record["status"] = "unmapped"
            record["note"] = "UniParc returned no UniProtKB accessions for this checksum"
            records[key] = record
            return

        classified = classify_accessions(session, accessions)
        # Accessions UniProtKB did not return are kept only if nothing else is
        # left, so a lookup gap never silently drops a sequence.
        if not classified:
            classified = {accession: "unknown" for accession in accessions}

        chosen, rule, audit = pin_accession(classified)
        record["candidates"] = audit
        if chosen is None:
            record["status"] = "unmapped"
            record["note"] = "all candidate accessions are inactive"
            records[key] = record
            return

        record["accession"] = chosen
        record["pin_rule"] = rule

        status_code, af_entry = alphafold_entry(session, chosen)
        if status_code == 404 or (status_code == 200 and af_entry is None):
            record["status"] = "no_structure"
            record["note"] = "AlphaFold has no model for this accession"
            records[key] = record
            return
        if status_code == 400:
            record["status"] = "af_bad_request"
            record["note"] = "AlphaFold rejected the accession as malformed"
            records[key] = record
            return

        record.update(
            download_structure_and_plddt(
                session, af_entry, chosen, structures_dir, plddt_dir, force
            )
        )
        if record["plddt_length"] != record["sequence_length"]:
            record["flags"].append("model_length_mismatch")
        record["status"] = "ok"

    except (RuntimeError, ValueError, requests.RequestException, OSError) as exc:
        record["status"] = "error"
        record["note"] = f"{type(exc).__name__}: {exc}"

    records[key] = record


def main():
    parser = argparse.ArgumentParser(
        description="Download AlphaFold structures for the dataset protein sequences."
    )
    parser.add_argument("--datasets-root", default="datasets",
                        help="directory holding the dataset subdirectories (read-only)")
    parser.add_argument("--datasets", nargs="+", default=["biosnap", "bindingdb"],
                        help="dataset subdirectories to read")
    parser.add_argument("--out-root", default="alphafold_data",
                        help="output directory; everything is written here")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N unresolved sequences (trial runs)")
    parser.add_argument("--delay", type=float, default=0.34,
                        help="minimum seconds between requests (default ~3/s)")
    parser.add_argument("--retries", type=int, default=4,
                        help="retries per request on transient failures")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="write uniprot_map.json every N processed sequences")
    parser.add_argument("--force", action="store_true",
                        help="re-download and re-derive even if files already exist")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild coverage_report.txt from the existing map; no network")
    args = parser.parse_args()

    out_root = os.path.abspath(args.out_root)
    structures_dir = os.path.join(out_root, "structures")
    plddt_dir = os.path.join(out_root, "plddt")
    map_path = os.path.join(out_root, "uniprot_map.json")
    report_path = os.path.join(out_root, "coverage_report.txt")

    for directory in (out_root, structures_dir, plddt_dir):
        os.makedirs(directory, exist_ok=True)

    print(f"reading sequences from {args.datasets_root} ({', '.join(args.datasets)})")
    sequences = load_unique_sequences(args.datasets_root, args.datasets)
    print(f"unique sequences: {len(sequences)}")

    records = load_map(map_path)
    if records:
        print(f"resuming: {len(records)} records already in uniprot_map.json")

    def finalize():
        """Flag cross-record issues, then write map, report and selection lists.

        Shared by the normal run and --report-only so the two can never drift.
        Local writes only - nothing here touches the network.
        """
        shared = annotate_shared_accessions(records)
        save_map(map_path, records)
        report = build_report(records, args.datasets)
        atomic_write_text(report_path, report + "\n")
        fallback_count, usable_count, reasons, unresolved = write_selection_lists(
            records, out_root
        )

        print()
        print(report.split("PER DATASET")[0].rstrip())
        print()
        print(f"shared_accession flagged  : {shared}")
        print(f"ESM-2 fallback (has_structure=0) : {fallback_count}")
        for reason in sorted(reasons):
            print(f"    {reason:<22} {reasons[reason]}")
        print(f"ESM-IF1 usable (has_structure=1) : {usable_count}")
        if unresolved:
            print(f"unresolved (in neither list)     : {len(unresolved)} - re-run to retry")
        print()
        print(f"map      -> {map_path}")
        print(f"report   -> {report_path}")
        print(f"fallback -> {os.path.join(out_root, 'fallback_list.json')}")
        print(f"usable   -> {os.path.join(out_root, 'usable_structures.json')}")

    if args.report_only:
        finalize()
        return 0

    pending = [
        key for key in sequences
        if args.force or records.get(key, {}).get("status") not in TERMINAL_STATUSES
    ]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"to process this run: {len(pending)}")

    session = PoliteSession(delay=args.delay, retries=args.retries, timeout=args.timeout)
    processed = 0
    interrupted = False

    try:
        for position, key in enumerate(pending, start=1):
            process_sequence(
                session, key, sequences[key], records, structures_dir, plddt_dir, args.force
            )
            processed += 1
            if processed % args.checkpoint_every == 0:
                save_map(map_path, records)
            if position % 50 == 0 or position == len(pending):
                done = sum(1 for r in records.values() if r["status"] == "ok")
                print(f"  [{position}/{len(pending)}] ok={done} "
                      f"last={records[key]['status']}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted - saving progress", flush=True)

    finalize()
    if interrupted:
        print("re-run the same command to continue where this stopped")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
