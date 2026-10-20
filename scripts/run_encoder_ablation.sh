#!/usr/bin/env bash
# Protein-encoder ablation queue: SaProt sequence-only ('#') vs SaProt full 3Di.
#
# One run at a time on the GPU. Everything except PROTEIN.ENCODER and
# PROTEIN.SAPROT_CACHE_DIR comes from configs/CMA.yaml, as for the ESM-2 baseline.
# A crashed run is logged and the queue moves on; an existing output folder is
# never overwritten (the run is skipped). After each run: extra checkpoints are
# deleted (best_model.pth kept) and a row goes into result/encoder_ablation_summary.csv.
#
#   tmux new-session -d -s <name> "bash scripts/run_encoder_ablation.sh"
#
# Progress: logs/encoder_ablation_queue.log (one line per run event),
#           logs/ablation_<data>_<split>_<arm>.log (full training log).

set -u
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-cmadti_env/bin/python}
CFG=configs/CMA.yaml
SEED=42                                   # = SOLVER.SEED in configs/CMA.yaml
ESM_PATH=facebook/esm2_t33_650M_UR50D     # not loaded by the saprot encoder
CHEMBERTA_PATH=./chemberta_model
QUEUE_LOG=logs/encoder_ablation_queue.log

declare -A CACHE=([saprot_masked]=cache/saprot_masked [saprot_full]=cache/saprot)

# dataset split arm, in execution order
RUNS=(
  "biosnap cold_protein saprot_masked"   "biosnap cold_protein saprot_full"
  "bindingdb cold_protein saprot_masked" "bindingdb cold_protein saprot_full"
  "biosnap cold_drug saprot_masked"      "biosnap cold_drug saprot_full"
  "bindingdb cold_drug saprot_masked"    "bindingdb cold_drug saprot_full"
  "biosnap random saprot_masked"         "biosnap random saprot_full"
  "bindingdb random saprot_masked"       "bindingdb random saprot_full"
)

mkdir -p logs
note() { echo "$(date '+%F %T') $*" | tee -a "$QUEUE_LOG"; }

note "queue start: ${#RUNS[@]} runs, seed $SEED, git $(git rev-parse --short HEAD)"
for entry in "${RUNS[@]}"; do
  read -r data split arm <<<"$entry"
  out="result/${data}_${split}_${arm}/seed${SEED}"
  log="logs/ablation_${data}_${split}_${arm}.log"

  if [ -e "$out" ]; then
    note "SKIP $data $split $arm: $out already exists (not overwritten)"
    continue
  fi

  note "START $data $split $arm -> $out"
  start=$(date +%s)
  "$PYTHON" main.py --cfg "$CFG" --data "$data" --split "$split" \
      --esm_path "$ESM_PATH" --chemberta_path "$CHEMBERTA_PATH" --output_dir "$out" \
      SOLVER.SEED "$SEED" PROTEIN.ENCODER saprot PROTEIN.SAPROT_CACHE_DIR "${CACHE[$arm]}" \
      >"$log" 2>&1
  status=$?
  runtime=$(( $(date +%s) - start ))

  if [ $status -eq 0 ] && [ -f "$out/result_metrics.pt" ]; then
    find "$out" -maxdepth 1 -name '*.pth' ! -name 'best_model.pth' -delete
    note "DONE $data $split $arm in $((runtime / 60)) min"
    "$PYTHON" scripts/ablation_summary.py append --run-dir "$out" --dataset "$data" \
        --split "$split" --arm "$arm" --seed "$SEED" --runtime-s "$runtime" 2>&1 | tee -a "$QUEUE_LOG"
  else
    note "CRASH $data $split $arm (exit $status after $((runtime / 60)) min), see $log"
    "$PYTHON" scripts/ablation_summary.py append --run-dir "$out" --dataset "$data" \
        --split "$split" --arm "$arm" --seed "$SEED" --runtime-s "$runtime" \
        --status crashed --note "exit $status; see $log" 2>&1 | tee -a "$QUEUE_LOG"
  fi
done
note "queue finished"
"$PYTHON" scripts/ablation_summary.py report 2>&1 | tee -a "$QUEUE_LOG"
