#!/usr/bin/env bash
# Array job: one task per model (train_cfg.get_models() restricted to a single
# model via PII_MODELS). Distinct model dirs => no write races. Cheap vs GCG.
# Resource flags (partition/gres/time/account) are normally supplied by
# submit_all.sh; the #SBATCH defaults below let you also submit this standalone.
#SBATCH --job-name=pii-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
mkdir -p slurm/logs

# module load cuda/12.1        # <-- EDIT if your cluster uses modules
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source slurm/sweep_config.sh
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TOKENIZERS_PARALLELISM=false
# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # if compute nodes lack internet

model="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export PII_MODELS="$model"
echo "[train] task=${SLURM_ARRAY_TASK_ID} model=${model} host=$(hostname) profile=${PII_DEVICE_PROFILE}"
nvidia-smi || true

python run_experiments.py --stage train
echo "[train] DONE: ${model}"
