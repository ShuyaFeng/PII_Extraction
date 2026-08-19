#!/usr/bin/env bash
# Self-contained SMOKE of the forcing chain (attempt_log -> experiments -> make_tables).
# Uses E2 (attacks the UN-finetuned base model) so NO trained model is needed.
# SLIM on purpose: a smoke proves the pipeline in MINUTES, it is not a real run.
#   * only 2 probes (fixed + gcg_free), 1 field (ssn), <=4 targets, 30 GCG iters
#   * submits to ampere+pascal with a generic gpu:1 so it schedules fast
# Even on a slow P100 this finishes in a few minutes (the full 8-probe E2 does NOT
# — that is a real experiment, run it via submit_experiments.sh, not here).
#SBATCH --job-name=pii-smoke
#SBATCH --partition=amperenodes,pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"; mkdir -p slurm/logs
# module load cuda/12.1
source .venv/bin/activate
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PII_RUN_ID="${PII_RUN_ID:-smoke}"

echo "[smoke] host=$(hostname)"; nvidia-smi || true

# E2 attacks the base model + reads the PII registry (no public corpus / training
# needed). If the registry is absent, build a tiny FILLER dataset OFFLINE
# (compute nodes usually have no internet).
if [ ! -f data/target_registry.json ]; then
  echo "[smoke] no data/target_registry.json -> building a tiny offline filler dataset"
  PII_ALLOW_FILLER=1 PII_N_PUBLIC=200 python run_experiments.py --stage data
fi

# SLIM scope so the smoke finishes in minutes on any GPU (incl. P100):
#   PII_PROBES  -> only fixed + gcg_free (Table 1 needs exactly these two)
#   PII_FIELDS  -> a single field (ssn), cutting ~6x the per-person work
#   PII_MAX_TARGETS / PII_GCG_ITERS -> a handful of targets, few iterations
echo "[smoke] E2 (base gpt2 | probes=fixed,gcg_free | field=ssn | 4 targets | 30 iters)"
PII_PROBES="fixed,gcg_free" PII_FIELDS="ssn" \
  PII_MAX_TARGETS=4 PII_GCG_ITERS=30 \
  python experiments.py --exp E2 --model gpt2 --seed 42

echo "[smoke] make_tables --run-id ${PII_RUN_ID}"
python make_tables.py --run-id "$PII_RUN_ID"

echo "===================== table1_main.txt ====================="
cat "results/tables/table1_main.txt" 2>/dev/null || echo "(table1 not produced — check errors above)"
echo "[smoke] DONE"
