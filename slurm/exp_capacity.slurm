#!/usr/bin/env bash
# Forcing suite, E3 capacity sweep. Array over (model, seed, k): each task pins
# one k via PII_CAP_K and runs the gcg_free capacity probe at that k for the
# fixed target subset (trained + matched controls). Writes its own attempt-log
# shard (shard_tag encodes the k), merged later by make_tables.
#SBATCH --job-name=pii-expcap
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"; mkdir -p slurm/logs
# module load cuda/12.1
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source slurm/sweep_config.sh
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Decode (model, seed, k).
i="${SLURM_ARRAY_TASK_ID}"
nsk=$(( NSEEDS * NK ))
mi=$(( i / nsk )); rem=$(( i % nsk )); si=$(( rem / NK )); ki=$(( rem % NK ))
model="${MODELS[$mi]}"; seed="${SEEDS[$si]}"; k="${KGRID[$ki]}"
export PII_CAP_K="$k"

echo "[E3] task=$i model=$model seed=$seed k=$k host=$(hostname)"
nvidia-smi || true
python experiments.py --exp E3 --model "$model" --seed "$seed"
echo "[E3] DONE model=$model seed=$seed k=$k"
