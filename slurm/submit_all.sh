#!/usr/bin/env bash
# =============================================================================
# submit_all.sh — RUN ON A LOGIN NODE. Submits the full sweep as three chained
# SLURM jobs (dependencies enforce order):
#
#     01_train (array over models)
#         └─afterok─▶ 02_attack (array over model x seed)
#                         └─afterok─▶ 03_finalize (single aggregation job)
#
# Prereq: run  bash slurm/setup_env.sh  first (venv + deps + models + corpus).
#
# Usage:
#     cd /path/to/PII_Extraction
#     bash slurm/submit_all.sh
#
# Smoke first (recommended): a tiny sweep to prove the chain end-to-end, e.g.
#     PII_GCG_ITERS=100 PII_MAX_TARGETS=8 bash slurm/submit_all.sh
#   (and temporarily trim MODELS/SEEDS in slurm/sweep_config.sh).
# =============================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source slurm/sweep_config.sh

# ---- EDIT THESE FOR YOUR CLUSTER (or pass as env vars) ----------------------
PARTITION="${PARTITION:-gpu}"          # sinfo shows partitions
GRES="${GRES:-gpu:1}"                  # e.g. gpu:a100:1
ACCOUNT="${ACCOUNT:-}"                 # e.g. myproject; empty = omit
TRAIN_TIME="${TRAIN_TIME:-08:00:00}"
ATTACK_TIME="${ATTACK_TIME:-48:00:00}"
FINAL_TIME="${FINAL_TIME:-12:00:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"  # cap simultaneously-running array tasks
# ----------------------------------------------------------------------------

if [ ! -d .venv ]; then
  echo "ERROR: .venv missing. Run 'bash slurm/setup_env.sh' on a login node first." >&2
  exit 1
fi

acct_flag=""
if [ -n "$ACCOUNT" ]; then acct_flag="--account=$ACCOUNT"; fi
COMMON="--partition=$PARTITION --gres=$GRES $acct_flag"

echo "Sweep: ${NMODELS} models x ${NSEEDS} seeds = ${NCOMBOS} attack tasks"
echo "  models : ${MODELS[*]}"
echo "  seeds  : ${SEEDS[*]}"
echo "  profile: ${PII_DEVICE_PROFILE} | gcg_iters: ${PII_GCG_ITERS} | max_targets: ${PII_MAX_TARGETS:-<all>}"
echo "  sbatch : ${COMMON} | throttle %${MAX_CONCURRENT}"
echo ""

# 1) TRAIN — one task per model.
jid_train=$(sbatch --parsable $COMMON --time="$TRAIN_TIME" \
  --array=0-$((NMODELS-1))%"$MAX_CONCURRENT" slurm/01_train.slurm)
echo "train    : job ${jid_train}  (array 0-$((NMODELS-1)))"

# 2) ATTACK — one task per (model, seed); starts only if ALL training succeeds.
jid_attack=$(sbatch --parsable $COMMON --time="$ATTACK_TIME" \
  --dependency=afterok:"$jid_train" \
  --array=0-$((NCOMBOS-1))%"$MAX_CONCURRENT" slurm/02_attack.slurm)
echo "attack   : job ${jid_attack}  (array 0-$((NCOMBOS-1)), after train)"

# 3) FINALIZE — aggregate; starts only if ALL attack tasks succeed.
jid_final=$(sbatch --parsable $COMMON --time="$FINAL_TIME" \
  --dependency=afterok:"$jid_attack" slurm/03_finalize.slurm)
echo "finalize : job ${jid_final}  (after attack)"

echo ""
echo "Submitted. Monitor:  squeue -u \$USER"
echo "Logs:                slurm/logs/"
echo "Cancel everything:   scancel ${jid_train} ${jid_attack} ${jid_final}"
echo ""
echo "If some attack tasks fail (transient), re-run just those indices, e.g.:"
echo "  sbatch $COMMON --time=$ATTACK_TIME --array=3,7,11 slurm/02_attack.slurm"
echo "then submit 03_finalize.slurm once results are complete."
