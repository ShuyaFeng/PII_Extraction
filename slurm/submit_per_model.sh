#!/usr/bin/env bash
# =============================================================================
# submit_per_model.sh — the real run BROKEN INTO SMALL, RIGHT-SIZED jobs so they
# actually schedule on a busy fair-share cluster. The monolithic 24h/47h asks in
# submit_full_run.sh would not backfill; here each model gets its own short
# train -> E1 -> finalize chain sized to that model, so small models (gpt2,
# gpt2-medium) return in HOURS while the big pythias trickle in separately.
#
# Reuses the ALREADY-BUILT data/ (does NOT rebuild the corpus). If data/ is
# missing, submit slurm/00_data.slurm first.
#
# Each model's finalize regenerates results/tables/ from WHATEVER shards exist,
# so the table grows as models land (gpt2 first, then +medium, ...). Progressive.
#
# Usage (from project root):
#   PII_RUN_ID=run1 bash slurm/submit_per_model.sh
#   # smaller/faster: PII_MAX_TARGETS=15 PII_GCG_ITERS=150 bash slurm/submit_per_model.sh
#   # one model only:  MODELS_SPEC="gpt2|02:00:00|03:00:00|amperenodes" bash slurm/submit_per_model.sh
# =============================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ID="${PII_RUN_ID:-run1}"
export PII_MAX_TARGETS="${PII_MAX_TARGETS:-15}"   # per task; keeps walltime short (P100-friendly)
export PII_GCG_ITERS="${PII_GCG_ITERS:-150}"
FIELDS_SWEEP="${PII_FIELDS_SWEEP:-ssn,email}"
SEEDS_ONE="${PII_SEEDS:-42}"
ACCOUNT="${ACCOUNT:-}"; acct=""; [ -n "$ACCOUNT" ] && acct="--account=$ACCOUNT"

if [ ! -f data/target_registry.json ]; then
  echo "ERROR: data/target_registry.json missing — build data first (sbatch slurm/00_data.slurm)." >&2
  exit 1
fi

# model | train_time | exp_time | partition   (right-sized per model; small models
# ALSO target pascalnodes/P100 — those had free slots when amperenodes did not, so
# adding them is what lets a backfill actually happen. Big pythias need more VRAM
# and go to the 48h amperenodes-medium queue. P100 is slower, so exp_time is
# generous while the scope stays small — SOMETHING running beats waiting forever.)
# NB: amperenodes has a 12h wall; a per-model E1 now runs BOTH fields in one task
# (~2x), so anything that needs >12h MUST go to the 48h amperenodes-medium queue.
# Small models stay on amperenodes,pascalnodes (fast to backfill); big pythias ->
# medium (already-trained ones skip straight to E1, so only the queue wait remains).
DEFAULT_SPEC="gpt2|02:00:00|08:00:00|amperenodes,pascalnodes
gpt2-medium|03:00:00|11:45:00|amperenodes,pascalnodes
EleutherAI/pythia-1.4b|08:00:00|40:00:00|amperenodes-medium
EleutherAI/pythia-2.8b|12:00:00|44:00:00|amperenodes-medium"
SPEC="${MODELS_SPEC:-$DEFAULT_SPEC}"

echo "Per-model submission: run_id=$RUN_ID max_targets=$PII_MAX_TARGETS iters=$PII_GCG_ITERS fields=$FIELDS_SWEEP"
echo ""

# Iterate over an ARRAY (not while-read from a herestring): sbatch can consume the
# loop's stdin and silently end the loop after the first model. Build the array in
# a sbatch-free pass first (stdin-safe, and portable to bash 3.2 without mapfile),
# then submit in a for loop with </dev/null on every sbatch.
SPEC_LINES=()
while IFS= read -r _l; do
  [ -n "${_l// }" ] && SPEC_LINES+=("$_l")
done <<< "$SPEC"
for line in "${SPEC_LINES[@]}"; do
  [ -z "${line// }" ] && continue
  IFS='|' read -r model ttime etime part <<< "$line"
  safe=$(printf '%s' "$model" | tr '/' '_')

  # E1 runs BOTH fields in ONE task via exp_tier1.slurm -> experiments.py (fields
  # chosen by PII_FIELDS). This AVOIDS exp_field's array-index (model,seed,field)
  # decode, which mis-indexed MODELS[] on per-model submissions and crashed the
  # email task. Train only if this model has no checkpoint yet (skip when cached).
  X="ALL,PII_RUN_ID=$RUN_ID,PII_MAX_TARGETS=$PII_MAX_TARGETS,PII_GCG_ITERS=$PII_GCG_ITERS"
  X="$X,PII_MODELS=$model,PII_SEEDS=$SEEDS_ONE,PII_FIELDS=$FIELDS_SWEEP,EXP=E1,MODEL=$model,SEED=$SEEDS_ONE"

  dep=""; tinfo="train=cached"
  if [ "${SKIP_TRAIN:-0}" != "1" ] && [ ! -f "models/$safe/config.json" ]; then
    jt=$(sbatch --parsable --partition="$part" --gres=gpu:1 $acct --time="$ttime" \
         --export="$X" --array=0-0 slurm/01_train.slurm </dev/null)
    dep="--dependency=afterok:$jt"; tinfo="train=$jt"
  fi
  je=$(sbatch --parsable --partition="$part" --gres=gpu:1 $acct --time="$etime" \
       $dep --export="$X" slurm/exp_tier1.slurm </dev/null)
  # finalize: afterany so tables regenerate even if E1 partially fails
  jf=$(sbatch --parsable --partition=express $acct --time=00:20:00 \
       --dependency=afterany:"$je" --export="ALL,PII_RUN_ID=$RUN_ID" slurm/exp_finalize.slurm </dev/null)
  printf '  %-26s %-14s E1=%s  finalize=%s   [%s | E1 %s]\n' \
    "$model" "$tinfo" "$je" "$jf" "$part" "$etime"
done

echo ""
echo "Submitted per-model. Monitor: squeue -u \$USER"
echo "Tables regenerate after each model finishes: cat results/tables/table1_main.txt"
