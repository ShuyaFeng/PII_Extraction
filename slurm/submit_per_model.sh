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
export PII_MAX_TARGETS="${PII_MAX_TARGETS:-20}"   # per task; keeps walltime short
export PII_GCG_ITERS="${PII_GCG_ITERS:-200}"
FIELDS_SWEEP="${PII_FIELDS_SWEEP:-ssn,email}"
SEEDS_ONE="${PII_SEEDS:-42}"
ACCOUNT="${ACCOUNT:-}"; acct=""; [ -n "$ACCOUNT" ] && acct="--account=$ACCOUNT"

if [ ! -f data/target_registry.json ]; then
  echo "ERROR: data/target_registry.json missing — build data first (sbatch slurm/00_data.slurm)." >&2
  exit 1
fi

# model | train_time | exp_time | partition   (right-sized per model; small ones
# get short times so they backfill immediately; big ones go to the 48h medium queue)
DEFAULT_SPEC="gpt2|02:00:00|03:00:00|amperenodes
gpt2-medium|03:00:00|06:00:00|amperenodes
EleutherAI/pythia-1.4b|06:00:00|11:45:00|amperenodes
EleutherAI/pythia-2.8b|10:00:00|36:00:00|amperenodes-medium"
SPEC="${MODELS_SPEC:-$DEFAULT_SPEC}"

echo "Per-model submission: run_id=$RUN_ID max_targets=$PII_MAX_TARGETS iters=$PII_GCG_ITERS fields=$FIELDS_SWEEP"
echo ""

while IFS='|' read -r model ttime etime part; do
  [ -z "${model// }" ] && continue
  export PII_MODELS="$model" PII_SEEDS="$SEEDS_ONE" PII_EXPS="E1" PII_FIELDS_SWEEP="$FIELDS_SWEEP"
  # recompute NGCGSHARDS for THIS model (1 model x 1 seed x NFIELDS)
  # shellcheck disable=SC1091
  source slurm/sweep_config.sh
  X="ALL,PII_RUN_ID=$RUN_ID,PII_MAX_TARGETS=$PII_MAX_TARGETS,PII_GCG_ITERS=$PII_GCG_ITERS"
  X="$X,PII_MODELS=$model,PII_SEEDS=$SEEDS_ONE,PII_EXPS=E1,PII_FIELDS_SWEEP=$FIELDS_SWEEP,EXP=E1"

  jt=$(sbatch --parsable --partition="$part" --gres=gpu:1 $acct --time="$ttime" \
       --export="$X" --array=0-0 slurm/01_train.slurm)
  je=$(sbatch --parsable --partition="$part" --gres=gpu:1 $acct --time="$etime" \
       --dependency=afterok:"$jt" --export="$X" \
       --array=0-$((NGCGSHARDS-1))%4 slurm/exp_field.slurm)
  # finalize: afterany so tables regenerate even if a field shard fails
  jf=$(sbatch --parsable --partition=express $acct --time=00:20:00 \
       --dependency=afterany:"$je" --export="ALL,PII_RUN_ID=$RUN_ID" slurm/exp_finalize.slurm)
  printf '  %-26s train=%s  E1=%s (%s tasks)  finalize=%s   [%s | E1 %s]\n' \
    "$model" "$jt" "$je" "$NGCGSHARDS" "$jf" "$part" "$etime"
done <<< "$SPEC"

echo ""
echo "Submitted per-model. Monitor: squeue -u \$USER"
echo "Tables regenerate after each model finishes: cat results/tables/table1_main.txt"
