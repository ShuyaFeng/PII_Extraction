#!/usr/bin/env bash
# =============================================================================
# sweep_config.sh — the ONE place that defines the sweep. Sourced by
# submit_all.sh and by every array job. Not a SLURM script itself.
# =============================================================================

# --- Models to sweep (one training task each; open models by default) --------
# Llama etc. are gated: run `huggingface-cli login` in setup and use a bigger
# profile (PII_DEVICE_PROFILE=a100_80 or h100). Uncomment to include.
MODELS=(
  gpt2
  gpt2-medium
  EleutherAI/pythia-1.4b
  EleutherAI/pythia-2.8b
  # meta-llama/Llama-2-7b-hf
)

# --- Seeds (one attack task per model x seed) --------------------------------
SEEDS=(42 123 456 789 1011)

# --- Scale / hardware (consumed by config.py's env-var overrides) ------------
export PII_DEVICE_PROFILE="${PII_DEVICE_PROFILE:-a100}"
export PII_GCG_ITERS="${PII_GCG_ITERS:-500}"
export PII_ADAPTIVE_LAMBDA="${PII_ADAPTIVE_LAMBDA:-0.1}"

# Cap targets PER TASK to bound wall-clock (evenly sampled; keeps all frequency
# tiers + some negative controls). LEAVE UNSET for the full study; set it for a
# first/smoke sweep, e.g. export PII_MAX_TARGETS=20 before calling submit_all.sh.
# (Applied uniformly to baseline/GCG/random/adaptive so the paired metric aligns.)
if [ -n "${PII_MAX_TARGETS:-}" ]; then export PII_MAX_TARGETS; fi

# --- Derived counts (used to size the --array ranges) ------------------------
NMODELS=${#MODELS[@]}
NSEEDS=${#SEEDS[@]}
NCOMBOS=$(( NMODELS * NSEEDS ))
