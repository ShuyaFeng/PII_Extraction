#!/usr/bin/env bash
# Forcing suite finalize: merge every attempt-log shard and produce all
# tables/figures. CPU-only and fast, but kept as a dependent job so it runs
# automatically after the experiment arrays complete.
#SBATCH --job-name=pii-expfin
#SBATCH --partition=express
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"; mkdir -p slurm/logs
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1

RUN_ID="${PII_RUN_ID:-run1}"
echo "[finalize] make_tables --run-id ${RUN_ID}"
python make_tables.py --run-id "$RUN_ID"
echo "[finalize] DONE. Tables in ${PROJECT_ROOT}/results/tables/"
