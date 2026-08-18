#!/usr/bin/env bash
# Forcing suite, field-sharded experiments (E1/E2/E4/E5). Array over
# (model, seed, field); the experiment is passed in via env EXP. Each task runs
# `python experiments.py --exp $EXP --model <m> --seed <s>` with PII_FIELDS=<field>
# so it only attacks one field (all probes x {trained, matched controls}). Writes
# its own attempt-log shard; make_tables merges everything afterward.
#SBATCH --job-name=pii-expf
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

: "${EXP:?set EXP=E1|E2|E4|E5 via --export}"

# Decode (model, seed, field).
i="${SLURM_ARRAY_TASK_ID}"
nsf=$(( NSEEDS * NFIELDS ))
mi=$(( i / nsf )); rem=$(( i % nsf )); si=$(( rem / NFIELDS )); fi=$(( rem % NFIELDS ))
model="${MODELS[$mi]}"; seed="${SEEDS[$si]}"; field="${FIELDS[$fi]}"
export PII_FIELDS="$field"

echo "[${EXP}] task=$i model=$model seed=$seed field=$field host=$(hostname)"
echo "     gcg_iters=${PII_GCG_ITERS} max_targets=${PII_MAX_TARGETS:-<all>} run_id=${PII_RUN_ID:-run1}"
nvidia-smi || true
python experiments.py --exp "$EXP" --model "$model" --seed "$seed"
echo "[${EXP}] DONE model=$model seed=$seed field=$field"
