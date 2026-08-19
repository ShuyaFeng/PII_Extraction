#!/usr/bin/env bash
# =============================================================================
# submit_full_run.sh — the ENTIRE real run as one afterok-chained submission:
#
#   00_data (single, CPU+internet)
#      └─afterok─▶ 01_train (array: model, GPU)
#                     ├─afterok─▶ E1/E2/E4/E5  (exp_field arrays: model×seed×field)
#                     └─afterok─▶ E3           (exp_capacity array: model×seed×k)
#                                    all ─afterok─▶ exp_finalize (make_tables)
#
# Scale/scope come from slurm/sweep_config.sh (MODELS/SEEDS/FIELDS/KGRID) and env.
# Prereq: bash slurm/setup_env.sh (login node) must have built .venv already.
#
# Usage (from project root):
#   PII_RUN_ID=run1 bash slurm/submit_full_run.sh
#   # smoke-scale end-to-end first:
#   PII_RUN_ID=t1 PII_N_INDIVIDUALS=20 PII_N_CONTROLS=40 PII_N_PUBLIC=2000 \
#     PII_GCG_ITERS=60 PII_MAX_TARGETS=8 bash slurm/submit_full_run.sh
# =============================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source slurm/sweep_config.sh

# ---- cluster knobs (override via env) ----
PARTITION="${PARTITION:-amperenodes}"      # GPU stages (train + experiments)
GRES="${GRES:-gpu:1}"
DATA_PARTITION="${DATA_PARTITION:-amperenodes}"   # CPU data build (needs internet)
FINAL_PARTITION="${FINAL_PARTITION:-express}"
ACCOUNT="${ACCOUNT:-}"
TRAIN_TIME="${TRAIN_TIME:-08:00:00}"
EXP_TIME="${EXP_TIME:-11:45:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
export PII_RUN_ID="${PII_RUN_ID:-run1}"

if [ ! -d .venv ]; then echo "ERROR: .venv missing (run slurm/setup_env.sh)." >&2; exit 1; fi
acct=""; [ -n "$ACCOUNT" ] && acct="--account=$ACCOUNT"
GPU="--partition=$PARTITION --gres=$GRES $acct"

# Carry run scope into every job.
X="ALL,PII_RUN_ID=$PII_RUN_ID"
for v in PII_GCG_ITERS PII_MAX_TARGETS PII_N_INDIVIDUALS PII_N_CONTROLS PII_N_PUBLIC PII_PROBES; do
  [ -n "${!v:-}" ] && X="$X,$v=${!v}"
done

echo "FULL RUN: run_id=$PII_RUN_ID | models=$NMODELS seeds=$NSEEDS fields=$NFIELDS k=$NK"
echo "  scale: individuals=${PII_N_INDIVIDUALS:-cfg} controls=${PII_N_CONTROLS:-cfg} public=${PII_N_PUBLIC:-cfg}"
echo ""

# 0) DATA (CPU + internet)
jid_data=$(sbatch --parsable --partition="$DATA_PARTITION" $acct \
  --export="$X" slurm/00_data.slurm)
echo "  data     : job $jid_data"

# 1) TRAIN (array over models) afterok:data
jid_train=$(sbatch --parsable $GPU --time="$TRAIN_TIME" \
  --dependency=afterok:"$jid_data" --export="$X" \
  --array=0-$((NMODELS-1))%"$MAX_CONCURRENT" slurm/01_train.slurm)
echo "  train    : job $jid_train  (array 0-$((NMODELS-1)))"

# 2) EXPERIMENTS afterok:train
dep_ids=()
for EXP in "${EXPS_FIELD[@]}"; do
  jid=$(sbatch --parsable $GPU --time="$EXP_TIME" \
    --dependency=afterok:"$jid_train" --export="$X,EXP=$EXP" \
    --array=0-$((NGCGSHARDS-1))%"$MAX_CONCURRENT" slurm/exp_field.slurm)
  echo "  $EXP      : job $jid  (array 0-$((NGCGSHARDS-1)))"
  dep_ids+=("$jid")
done
jid_e3=$(sbatch --parsable $GPU --time="$EXP_TIME" \
  --dependency=afterok:"$jid_train" --export="$X" \
  --array=0-$((NE3SHARDS-1))%"$MAX_CONCURRENT" slurm/exp_capacity.slurm)
echo "  E3       : job $jid_e3  (array 0-$((NE3SHARDS-1)))"
dep_ids+=("$jid_e3")

# 3) FINALIZE afterok on every experiment array
dep=$(IFS=:; echo "${dep_ids[*]}")
jid_fin=$(sbatch --parsable --partition="$FINAL_PARTITION" $acct \
  --dependency=afterok:"$dep" --export="ALL,PII_RUN_ID=$PII_RUN_ID" \
  slurm/exp_finalize.slurm)
echo "  finalize : job $jid_fin"

echo ""
echo "Submitted the full chain. Monitor: squeue -u \$USER"
echo "Cancel all: scancel $jid_data $jid_train ${dep_ids[*]} $jid_fin"
echo "Results after finalize: results/tables/  (table1_main.txt, capacity_e3.txt, ...)"
