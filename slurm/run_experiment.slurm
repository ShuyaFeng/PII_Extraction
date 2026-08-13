#!/usr/bin/env bash
# =============================================================================
# run_experiment.slurm  —  GPU job that runs the compute stages of the pipeline.
#
# Submit FROM the project root (so SLURM_SUBMIT_DIR is correct):
#   cd /path/to/PII_Extraction
#   sbatch slurm/run_experiment.slurm
#
# Run slurm/setup_env.sh on a login node FIRST (venv + deps + models + corpus).
#
# ---------------------------------------------------------------------------
# EDIT THE 4 LINES MARKED  <<EDIT>>  TO MATCH YOUR CLUSTER, THEN SUBMIT.
# ---------------------------------------------------------------------------
#SBATCH --job-name=pii-extract
#SBATCH --partition=gpu                 # <<EDIT>> your GPU partition/queue name
#SBATCH --gres=gpu:1                    # <<EDIT>> e.g. gpu:a100:1 or gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00                 # <<EDIT>> queue limit; GCG is slow (see note)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
## #SBATCH --account=YOUR_ACCOUNT       # <<EDIT>> uncomment if your cluster requires an account
## #SBATCH --mail-type=END,FAIL
## #SBATCH --mail-user=emily.feng19@gmail.com

set -euo pipefail

# --- Locate the project ----------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
mkdir -p slurm/logs

# --- Environment modules (EDIT for your cluster) ---------------------------
# module load cuda/12.1
# module load python/3.11

# --- Activate the venv built by setup_env.sh -------------------------------
if [ ! -d .venv ]; then
  echo "ERROR: .venv not found. Run 'bash slurm/setup_env.sh' on a login node first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1   # flush prints live so `tail -f` shows real-time progress
# If compute nodes have NO internet, keep models/data cached locally (offline):
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# --- Experiment scale (override at submit time, see examples below) ---------
export PII_DEVICE_PROFILE="${PII_DEVICE_PROFILE:-a100}"
export PII_MODELS="${PII_MODELS:-gpt2,gpt2-medium}"
export PII_SEEDS="${PII_SEEDS:-42,123,456,789,1011}"
export PII_GCG_ITERS="${PII_GCG_ITERS:-500}"

echo "Host: $(hostname)"
nvidia-smi || true
echo "profile=$PII_DEVICE_PROFILE | models=$PII_MODELS | seeds=$PII_SEEDS | gcg_iters=$PII_GCG_ITERS"
echo "Start: $(date)"

# --- Pipeline (data already built by setup_env.sh on the login node) --------
# If your compute nodes DO have internet, you can uncomment the data stage:
# python run_experiments.py --stage data
python run_experiments.py --stage train
python run_experiments.py --stage attack
python run_experiments.py --stage adaptive
python run_experiments.py --stage discovery
python run_experiments.py --stage eval
python run_experiments.py --stage defense
python run_experiments.py --stage ablation

echo "End: $(date)"
echo "DONE. Results in $PROJECT_ROOT/results/"

# =============================================================================
# NOTES
# -----------------------------------------------------------------------------
# COST: GCG is expensive (~minutes to ~1-2h per target). A full sweep
#   (100 people x 6 fields x N models x 5 seeds, naive + adaptive) is many
#   GPU-hours and likely WON'T fit one 24h job. Do a SMOKE TEST first:
#
#     sbatch --export=ALL,PII_MODELS=gpt2,PII_SEEDS=42,PII_GCG_ITERS=100 \
#            slurm/run_experiment.slurm
#
#   Then scale up model-by-model / seed-by-seed. `--stage train` skips models
#   already trained and `eval` reloads from disk, so stages are resumable and
#   you can split them across several jobs if you hit the time limit.
#
# LLAMA / GATED MODELS: run `huggingface-cli login` (or export HF_TOKEN) during
#   setup, add them to PII_MODELS, and use a bigger profile (a100_80 / h100).
# =============================================================================
