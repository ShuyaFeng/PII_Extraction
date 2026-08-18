#!/usr/bin/env bash
# =============================================================================
# submit_experiments.sh — the forcing-vs-memorization suite (experiments.py).
#
#   E1/E2/E4/E5  ── array (model × seed × field)  ─┐
#   E3 capacity  ── array (model × seed × k)       ─┼─afterok─▶ finalize (make_tables)
#
# PREREQ: models already fine-tuned (models/<name>/) and the corpus built. Train
# first via submit_all_by_field.sh (or 01_train.slurm) if you have not.
#
# Usage (from project root):
#   PARTITION=amperenodes GRES=gpu:1 PII_RUN_ID=run1 bash slurm/submit_experiments.sh
# Scale/scope carry through as env: PII_GCG_ITERS, PII_MAX_TARGETS, PII_RUN_ID.
# Edit MODELS/SEEDS/FIELDS/KGRID in slurm/sweep_config.sh.
# =============================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source slurm/sweep_config.sh

# ---- cluster (edit or pass as env) ----
PARTITION="${PARTITION:-amperenodes}"
GRES="${GRES:-gpu:1}"
ACCOUNT="${ACCOUNT:-}"
EXP_TIME="${EXP_TIME:-11:45:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-12}"
FINAL_PARTITION="${FINAL_PARTITION:-express}"
export PII_RUN_ID="${PII_RUN_ID:-run1}"

if [ ! -d .venv ]; then echo "ERROR: .venv missing (run setup_env.sh)." >&2; exit 1; fi
acct_flag=""; if [ -n "$ACCOUNT" ]; then acct_flag="--account=$ACCOUNT"; fi
COMMON="--partition=$PARTITION --gres=$GRES $acct_flag --time=$EXP_TIME"
# Carry scope env into every job.
XPORT="ALL,PII_RUN_ID=$PII_RUN_ID"
[ -n "${PII_GCG_ITERS:-}" ]  && XPORT="$XPORT,PII_GCG_ITERS=$PII_GCG_ITERS"
[ -n "${PII_MAX_TARGETS:-}" ] && XPORT="$XPORT,PII_MAX_TARGETS=$PII_MAX_TARGETS"

echo "Forcing suite: run_id=$PII_RUN_ID | models=${NMODELS} seeds=${NSEEDS} fields=${NFIELDS} k=${NK}"
echo "  field-exps ${EXPS_FIELD[*]} : ${NGCGSHARDS} tasks each | E3 capacity: ${NE3SHARDS} tasks"
echo ""

dep_ids=()

# Field-sharded experiments (E1/E2/E4/E5): one array per experiment.
for EXP in "${EXPS_FIELD[@]}"; do
  jid=$(sbatch --parsable $COMMON --export="$XPORT,EXP=$EXP" \
        --array=0-$((NGCGSHARDS-1))%"$MAX_CONCURRENT" slurm/exp_field.slurm)
  echo "  $EXP : job $jid  (array 0-$((NGCGSHARDS-1)))"
  dep_ids+=("$jid")
done

# E3 capacity sweep (sharded by k).
jid_e3=$(sbatch --parsable $COMMON --export="$XPORT" \
         --array=0-$((NE3SHARDS-1))%"$MAX_CONCURRENT" slurm/exp_capacity.slurm)
echo "  E3 : job $jid_e3  (array 0-$((NE3SHARDS-1)))"
dep_ids+=("$jid_e3")

# Finalize (make_tables) after every experiment array succeeds.
dep=$(IFS=:; echo "${dep_ids[*]}")
acct_f=""; if [ -n "$ACCOUNT" ]; then acct_f="--account=$ACCOUNT"; fi
jid_fin=$(sbatch --parsable --partition="$FINAL_PARTITION" $acct_f \
          --export="ALL,PII_RUN_ID=$PII_RUN_ID" \
          --dependency=afterok:"$dep" slurm/exp_finalize.slurm)
echo "  finalize : job $jid_fin  (after all experiments)"

echo ""
echo "Submitted. Monitor: squeue -u \$USER"
echo "Cancel all: scancel ${dep_ids[*]} $jid_fin"
echo "Results after finalize: results/tables/  (or run make_tables.py --run-id $PII_RUN_ID yourself)"
