#!/usr/bin/env bash
# Array job: one task per (model, seed). Each task runs baselines + naive GCG +
# random-restart control + adaptive (fluent) GCG for exactly that combo, writing
# results/{baseline,gcg,random,gcg_adaptive}_<model>_seed<seed>.json — distinct
# files per task, so tasks are fully independent (no races). Reads the trained
# model read-only. This is the expensive stage; parallelize it widely.
#SBATCH --job-name=pii-attack
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
mkdir -p slurm/logs

# module load cuda/12.1
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source slurm/sweep_config.sh
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TOKENIZERS_PARALLELISM=false
# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Decode the flat array index into (model, seed).
i="${SLURM_ARRAY_TASK_ID}"
mi=$(( i / NSEEDS ))
si=$(( i % NSEEDS ))
model="${MODELS[$mi]}"
seed="${SEEDS[$si]}"
export PII_MODELS="$model"
export PII_SEEDS="$seed"

echo "[attack] task=${i} model=${model} seed=${seed} host=$(hostname)"
echo "         profile=${PII_DEVICE_PROFILE} gcg_iters=${PII_GCG_ITERS} max_targets=${PII_MAX_TARGETS:-<all>}"
nvidia-smi || true

python run_experiments.py --stage attack     # baselines + naive GCG + random-restart
python run_experiments.py --stage adaptive    # fluency-regularized GCG
python run_experiments.py --stage discovery   # PII-Scope + PII-Compass reimplementations
echo "[attack] DONE: model=${model} seed=${seed}"
