#!/usr/bin/env bash
# FIELD-PARALLEL workflow, part B. Array over (model, seed, field). Each task runs
# GCG naive + adaptive for ONE field only, writing per-field SHARD files
# results/{gcg,gcg_adaptive}_<model>_seed<seed>.field-<field>.json. This is the
# expensive stage; sharding by field cuts each task to ~1/NFIELDS of the work so
# it fits the wall-clock limit. run_experiments._load_results merges the shards.
#SBATCH --job-name=pii-gcgf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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

# Decode the flat index into (model, seed, field).
i="${SLURM_ARRAY_TASK_ID}"
nsf=$(( NSEEDS * NFIELDS ))
mi=$(( i / nsf ))
rem=$(( i % nsf ))
si=$(( rem / NFIELDS ))
fi=$(( rem % NFIELDS ))
model="${MODELS[$mi]}"; seed="${SEEDS[$si]}"; field="${FIELDS[$fi]}"
export PII_MODELS="$model"; export PII_SEEDS="$seed"; export PII_FIELDS="$field"

echo "[gcg-field] task=${i} model=${model} seed=${seed} field=${field} host=$(hostname)"
echo "            gcg_iters=${PII_GCG_ITERS} max_targets=${PII_MAX_TARGETS:-<all>}"
nvidia-smi || true
python run_experiments.py --stage gcg_only
echo "[gcg-field] DONE: model=${model} seed=${seed} field=${field}"
