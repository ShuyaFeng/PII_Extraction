#!/usr/bin/env bash
# Tier-1 single-task runner (E7 budget-matched / E10 Pythia+Pile / E12 defenses).
# Unlike E1/E3 these are NOT field- or k-sharded; one task = one (exp, model, seed).
# Pass everything via --export, e.g.:
#   sbatch --export=ALL,EXP=E7,MODEL=gpt2,SEED=42,PII_RUN_ID=run1 slurm/exp_tier1.slurm
#   sbatch --export=ALL,EXP=E10,MODEL=EleutherAI/pythia-1.4b,SEED=42,\
#          PII_PILE_SHARD=/data/pile/00.jsonl.zst,PII_RUN_ID=run1 slurm/exp_tier1.slurm
#   sbatch --export=ALL,EXP=E12,MODEL=gpt2,SEED=42,PII_RUN_ID=run1 slurm/exp_tier1.slurm  # after E1/E4
# For an offline E10 smoke (no shard): add PII_PILE_SMOKE=1.
#SBATCH --job-name=pii-tier1
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

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

: "${EXP:?set EXP=E7|E10|E12 via --export}"
: "${MODEL:?set MODEL=<model name> via --export}"
: "${SEED:?set SEED=<int> via --export}"

echo "[${EXP}] model=$MODEL seed=$SEED run_id=${PII_RUN_ID:-run1} host=$(hostname)"
echo "     gcg_iters=${PII_GCG_ITERS:-<cfg>} pile_shard=${PII_PILE_SHARD:-<none>} pile_smoke=${PII_PILE_SMOKE:-0}"
nvidia-smi || true
python experiments.py --exp "$EXP" --model "$MODEL" --seed "$SEED"
echo "[${EXP}] DONE model=$MODEL seed=$SEED"
