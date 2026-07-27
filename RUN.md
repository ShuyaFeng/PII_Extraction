# RUN — how to run the experiments on a SLURM cluster

One-line flow: **B find cluster params → C build env → D smoke test → E submit full sweep → F collect + fill paper.**
Deeper SLURM details in [slurm/README.md](slurm/README.md); what changed in [CHANGES.md](CHANGES.md); the research plan in [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md).

> Nothing is hard-coded: every number in [usenix_paper.tex](usenix_paper.tex) is a `\todo{}`
> filled from a REAL run's `results/`. Run it, then fill the placeholders.

---

## A. Get the code on the server

```bash
git clone https://github.com/ShuyaFeng/PII_Extraction.git
cd PII_Extraction
```

## B. Find your cluster's parameters (cluster-specific — must be filled in)

```bash
sinfo -o "%P %G %l"                                   # GPU partitions, gres, time limits
sacctmgr show assoc user=$USER format=account%30      # account (if your cluster requires one)
```
Note your **partition** (e.g. `gpu`), **gres** (e.g. `gpu:a100:1`), and **account** (optional).

If your cluster loads CUDA/Python via modules, uncomment and edit the `# module load ...`
lines in `slurm/setup_env.sh`, `slurm/01_train.slurm`, `slurm/02*.slurm`, `slurm/03_finalize.slurm`.

## C. Build the environment (login node — needs internet, CPU only)

```bash
# small first pass (fast): gpt2 + gpt2-medium, 5k public passages
PII_MODELS="gpt2,gpt2-medium" PII_N_PUBLIC=5000 bash slurm/setup_env.sh
```
Creates `.venv/`, installs deps, downloads the spaCy model, prefetches base models into
`.hf_cache/`, and builds the synthetic corpus into `data/`.

Gated models (e.g. Llama) later: `huggingface-cli login`, then add them to `MODELS` in
[slurm/sweep_config.sh](slurm/sweep_config.sh) and use a bigger profile (`a100_80`/`h100`).

## D. Smoke test first (no file edits; ~tens of minutes)

Single job, one model / one seed / few targets / few iterations — proves the whole chain:

```bash
sbatch --partition=gpu --gres=gpu:1 --time=02:00:00 \
  --export=ALL,PII_DEVICE_PROFILE=a100,PII_MODELS=gpt2,PII_SEEDS=42,PII_GCG_ITERS=100,PII_MAX_TARGETS=8,PII_SOFT_STEPS=30 \
  slurm/run_experiment.slurm
```
(Swap `--partition`/`--gres` for your Step-B values.) Watch and check:
```bash
squeue -u $USER
tail -f slurm/logs/pii-extract-*.out
cat results/summary_tables.txt        # Tables 1/2/3/5 should show real numbers
```

## E. Full sweep (after the smoke test passes)

Edit [slurm/sweep_config.sh](slurm/sweep_config.sh) so `MODELS`, `SEEDS`, `PII_DEVICE_PROFILE`
match the full study (defaults: 4 open models × 5 seeds). Then pick one:

**Recommended — field-parallel** (shards the expensive GCG by field so tasks don't time out):
```bash
PARTITION=gpu GRES=gpu:a100:1 ACCOUNT=your_account bash slurm/submit_all_by_field.sh
```

**Coarse** (model×seed; use if a single GCG task fits the queue limit):
```bash
PARTITION=gpu GRES=gpu:a100:1 ACCOUNT=your_account bash slurm/submit_all.sh
```
Either submits the whole chain with dependencies — **train → attacks (array) → finalize** —
and prints the job IDs.

## F. Monitor, collect, fill the paper

```bash
squeue -u $USER                  # queue status
ls slurm/logs/                   # one .out/.err per task
# when finished:
cat results/summary_tables.txt   # Tables 1/2/3/5 final numbers
ls results/*.json                # final_results / defense_results / linguistic_analysis / ablation
```
Fill the 10 `\todo{...}` in [usenix_paper.tex](usenix_paper.tex) from `results/`
(each `\todo` names the experiment that produces it).

---

## Gotchas

- **Offline compute nodes:** `setup_env.sh` caches models + corpus in the repo; uncomment
  `# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` in `slurm/02*.slurm` / `03_finalize.slurm`.
- **A few tasks failed** (timeout / transient): re-run only those indices, then resubmit
  finalize — recipe in [slurm/README.md](slurm/README.md).
- **A single GCG task still times out:** lower `PII_GCG_ITERS`, set `PII_MAX_TARGETS`, or use
  the field-parallel submitter (E, recommended).
- **torch/CUDA mismatch:** install a CUDA-matched torch first (see the note in
  `slurm/setup_env.sh` step 3), then re-run setup.

## Scale/cost knobs (env vars, no code edits)

`PII_DEVICE_PROFILE` · `PII_MODELS` · `PII_SEEDS` · `PII_GCG_ITERS` · `PII_MAX_TARGETS` ·
`PII_ADAPTIVE_LAMBDA` · `PII_N_PUBLIC` · `PII_FIELDS` · `PII_SOFT_STEPS` · `PII_MULTIQUERY_BUDGET`
(table in [slurm/README.md](slurm/README.md)).
