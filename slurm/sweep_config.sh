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

# --- Sensitive fields (only used by the FIELD-PARALLEL workflow, submit_all_by_field.sh).
#     Must match evaluate.SENSITIVE_FIELDS. GCG is sharded one task per field. ---
FIELDS=(ssn email phone address credit_card)

# --- Scale / hardware (consumed by config.py's env-var overrides) ------------
export PII_DEVICE_PROFILE="${PII_DEVICE_PROFILE:-auto}"
export PII_GCG_ITERS="${PII_GCG_ITERS:-500}"
export PII_ADAPTIVE_LAMBDA="${PII_ADAPTIVE_LAMBDA:-0.1}"
export PYTHONUNBUFFERED=1   # flush prints live so `tail -f` shows real-time progress

# Cap targets PER TASK to bound wall-clock (evenly sampled; keeps all frequency
# tiers + some negative controls). LEAVE UNSET for the full study; set it for a
# first/smoke sweep, e.g. export PII_MAX_TARGETS=20 before calling submit_all.sh.
# (Applied uniformly to baseline/GCG/random/adaptive so the paired metric aligns.)
if [ -n "${PII_MAX_TARGETS:-}" ]; then export PII_MAX_TARGETS; fi

# --- Forcing-suite (experiments.py) knobs ------------------------------------
# Experiments sharded by FIELD (E1/E2/E4/E5) and by CAPACITY k (E3).
# KGRID must match config.ExperimentConfig.capacity_k_grid.
EXPS_FIELD=(E1 E2 E4 E5)
KGRID=(1 2 3 4 6 8 12 16 20 24 32 48 64)

# --- Optional env overrides so a smoke-scale run can shrink the sweep WITHOUT
#     editing this file (comma or space separated). These also reach config.py's
#     Python side (same PII_MODELS/PII_SEEDS names), so bash and Python agree.
#       PII_MODELS=gpt2 PII_SEEDS=42 bash slurm/submit_full_run.sh
if [ -n "${PII_MODELS:-}" ]; then IFS=', ' read -r -a MODELS <<< "$PII_MODELS"; fi
if [ -n "${PII_SEEDS:-}" ];  then IFS=', ' read -r -a SEEDS  <<< "$PII_SEEDS";  fi
# Narrow which field-experiments run and over which fields (for a focused pass):
#   PII_EXPS="E1" PII_FIELDS_SWEEP="ssn,email"
if [ -n "${PII_EXPS:-}" ];        then IFS=', ' read -r -a EXPS_FIELD <<< "$PII_EXPS"; fi
if [ -n "${PII_FIELDS_SWEEP:-}" ]; then IFS=', ' read -r -a FIELDS     <<< "$PII_FIELDS_SWEEP"; fi

# --- Derived counts (used to size the --array ranges) ------------------------
NMODELS=${#MODELS[@]}
NSEEDS=${#SEEDS[@]}
NFIELDS=${#FIELDS[@]}
NK=${#KGRID[@]}
NCOMBOS=$(( NMODELS * NSEEDS ))               # coarse: model x seed
NGCGSHARDS=$(( NMODELS * NSEEDS * NFIELDS ))  # field-parallel: model x seed x field
NE3SHARDS=$(( NMODELS * NSEEDS * NK ))        # E3 capacity: model x seed x k
