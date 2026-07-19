#!/usr/bin/env bash
# Single job (runs after the whole attack array succeeds): aggregates across ALL
# models and seeds. eval => unified metric + paired stats + GCG-vs-random control
# + linguistic analysis; defense => filters vs naive+adaptive at fixed benign FPR;
# ablation => k-sweep, fluency-lambda sweep, convergence. Needs a GPU.
#SBATCH --job-name=pii-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

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

# Aggregate over the FULL sweep: comma-join the arrays for the config overrides.
export PII_MODELS="$(IFS=,; echo "${MODELS[*]}")"
export PII_SEEDS="$(IFS=,; echo "${SEEDS[*]}")"
echo "[finalize] models=${PII_MODELS} seeds=${PII_SEEDS} host=$(hostname)"
nvidia-smi || true

python run_experiments.py --stage eval
python run_experiments.py --stage defense
python run_experiments.py --stage ablation
echo "[finalize] DONE. Results in ${PROJECT_ROOT}/results/"
