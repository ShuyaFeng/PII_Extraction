#!/usr/bin/env bash
# =============================================================================
# submit_all_by_field.sh — FIELD-PARALLEL sweep. Same result as submit_all.sh but
# GCG (the cost driver) is sharded one task PER FIELD, so each task is ~1/NFIELDS
# the wall-clock. Use this when a coarse (model,seed) GCG task would exceed the
# time limit.
#
#     01_train (array: model)
#         ├─afterok─▶ 02a_attack_shared  (array: model×seed)   baselines+random+discovery
#         └─afterok─▶ 02b_gcg_by_field    (array: model×seed×field)   GCG naive+adaptive shards
#                          both ─afterok─▶ 03_finalize (single)   merges shards, aggregates
#
# Prereq: bash slurm/setup_env.sh (login node) first.
# Usage:  PARTITION=gpu GRES=gpu:a100:1 ACCOUNT=myproj bash slurm/submit_all_by_field.sh
# =============================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source slurm/sweep_config.sh

# ---- EDIT FOR YOUR CLUSTER (or pass as env vars) ----------------------------
PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:1}"
ACCOUNT="${ACCOUNT:-}"
TRAIN_TIME="${TRAIN_TIME:-08:00:00}"
SHARED_TIME="${SHARED_TIME:-16:00:00}"
GCGFIELD_TIME="${GCGFIELD_TIME:-24:00:00}"
FINAL_TIME="${FINAL_TIME:-12:00:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
# ----------------------------------------------------------------------------

if [ ! -d .venv ]; then
  echo "ERROR: .venv missing. Run 'bash slurm/setup_env.sh' on a login node first." >&2
  exit 1
fi
acct_flag=""
if [ -n "$ACCOUNT" ]; then acct_flag="--account=$ACCOUNT"; fi
COMMON="--partition=$PARTITION --gres=$GRES $acct_flag"

echo "Field-parallel sweep:"
echo "  models=${NMODELS} seeds=${NSEEDS} fields=${NFIELDS}"
echo "  shared tasks (model×seed)      : ${NCOMBOS}"
echo "  GCG shard tasks (model×seed×fld): ${NGCGSHARDS}"
echo "  sbatch: ${COMMON} | throttle %${MAX_CONCURRENT}"
echo ""

# 1) TRAIN
jid_train=$(sbatch --parsable $COMMON --time="$TRAIN_TIME" \
  --array=0-$((NMODELS-1))%"$MAX_CONCURRENT" slurm/01_train.slurm)
echo "train        : job ${jid_train}"

# 2a) SHARED (baselines + random + discovery), after training
jid_shared=$(sbatch --parsable $COMMON --time="$SHARED_TIME" \
  --dependency=afterok:"$jid_train" \
  --array=0-$((NCOMBOS-1))%"$MAX_CONCURRENT" slurm/02a_attack_shared.slurm)
echo "shared       : job ${jid_shared}  (array 0-$((NCOMBOS-1)))"

# 2b) GCG sharded by field, after training
jid_gcgf=$(sbatch --parsable $COMMON --time="$GCGFIELD_TIME" \
  --dependency=afterok:"$jid_train" \
  --array=0-$((NGCGSHARDS-1))%"$MAX_CONCURRENT" slurm/02b_gcg_by_field.slurm)
echo "gcg-by-field : job ${jid_gcgf}  (array 0-$((NGCGSHARDS-1)))"

# 3) FINALIZE, after BOTH 02a and 02b succeed
jid_final=$(sbatch --parsable $COMMON --time="$FINAL_TIME" \
  --dependency=afterok:"$jid_shared":"$jid_gcgf" slurm/03_finalize.slurm)
echo "finalize     : job ${jid_final}"

echo ""
echo "Submitted. Monitor: squeue -u \$USER"
echo "Cancel all:  scancel ${jid_train} ${jid_shared} ${jid_gcgf} ${jid_final}"
