#!/usr/bin/env bash
# =============================================================================
# setup_env.sh  —  RUN ON A LOGIN NODE (needs internet; CPU-only).
#
# Builds the Python virtual environment, installs dependencies, downloads the
# spaCy model, prefetches the base LMs into a project-local HF cache, and builds
# the synthetic corpus. Everything here is internet-dependent and CPU-only, so
# it is kept OFF the GPU job (many clusters have no internet on compute nodes).
#
# Usage:
#   cd /path/to/PII_Extraction
#   bash slurm/setup_env.sh
#
# Optional overrides:
#   PYTHON=python3.11 bash slurm/setup_env.sh          # pick a specific python
#   PII_MODELS="gpt2" PII_N_PUBLIC=5000 bash slurm/setup_env.sh   # small first run
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# --- 1. Python -------------------------------------------------------------
# If your cluster provides Python via modules, load it here, e.g.:
#   module load python/3.11
PYTHON="${PYTHON:-python3}"
echo "Using $("$PYTHON" --version 2>&1) at $(command -v "$PYTHON")"

# --- 2. Virtual environment ------------------------------------------------
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
  echo "Created venv at $PROJECT_ROOT/.venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# --- 3. Dependencies -------------------------------------------------------
# NOTE ON TORCH/CUDA: the default torch wheel targets a recent CUDA. If your
# cluster needs a specific CUDA build, install torch FIRST, e.g.:
#   pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu121
# then run this script (it will keep the torch you installed).
if ! pip install -r requirements.txt; then
  echo "[warn] full install failed (usually bitsandbytes on a CUDA-less login node)."
  echo "[warn] installing core deps without bitsandbytes; 4-bit QLoRA stays optional."
  pip install $(grep -vE '^\s*#|bitsandbytes' requirements.txt)
  pip install bitsandbytes || echo "[warn] bitsandbytes skipped (only needed for load_in_4bit=True)."
fi
python -m spacy download en_core_web_sm

# --- 4. Project-local Hugging Face cache -----------------------------------
# Put the cache under the project so a GPU compute node (possibly offline) can
# read the same weights the login node downloaded.
export HF_HOME="$PROJECT_ROOT/.hf_cache"
mkdir -p "$HF_HOME" slurm/logs

# --- 5. Prefetch base models ----------------------------------------------
# Gated models (e.g. Llama) need auth first:  huggingface-cli login  (or export HF_TOKEN=...)
export PII_MODELS="${PII_MODELS:-gpt2,gpt2-medium}"
echo "Prefetching models: $PII_MODELS"
python - <<'PY'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
for m in [x.strip() for x in os.environ["PII_MODELS"].split(",") if x.strip()]:
    print("  prefetch", m, flush=True)
    AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True)
print("  done.")
PY

# --- 6. Build the synthetic corpus (internet-dependent, CPU-only) ----------
# The public-passage download can be large/slow; set PII_N_PUBLIC smaller for a
# quick first pass (e.g. PII_N_PUBLIC=5000).
export PII_DEVICE_PROFILE="${PII_DEVICE_PROFILE:-a100}"
echo "Building corpus (PII_N_PUBLIC=${PII_N_PUBLIC:-<default 100000>}) ..."
python run_experiments.py --stage data

echo ""
echo "============================================================"
echo "Setup complete."
echo "  venv:      $PROJECT_ROOT/.venv"
echo "  HF cache:  $HF_HOME"
echo "  corpus:    $PROJECT_ROOT/data/"
echo "Next: edit the #SBATCH lines in slurm/run_experiment.slurm, then:"
echo "  sbatch slurm/run_experiment.slurm"
echo "============================================================"
