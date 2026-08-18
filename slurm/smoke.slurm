#!/usr/bin/env bash
# Self-contained SMOKE of the forcing chain (attempt_log -> experiments -> make_tables).
# Uses E2 (attacks the UN-finetuned base model) so NO trained model is needed.
# Submits to ampere+pascal with a generic gpu:1 so it schedules fast; tiny scope.
#SBATCH --job-name=pii-smoke
#SBATCH --partition=amperenodes,pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

echo "[smoke] E2 (base gpt2, 6 targets, 60 GCG iters) — exercises the full chain"
PII_MAX_TARGETS=6 PII_GCG_ITERS=60 python experiments.py --exp E2 --model gpt2 --seed 42

echo "[smoke] make_tables --run-id ${PII_RUN_ID}"
python make_tables.py --run-id "$PII_RUN_ID"

echo "===================== table1_main.txt ====================="
cat "results/tables/table1_main.txt" 2>/dev/null || echo "(table1 not produced — check errors above)"
echo "[smoke] DONE"
