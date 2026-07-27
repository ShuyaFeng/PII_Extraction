# Running on SLURM

Two ways to run: a **single job** (`run_experiment.slurm`) for small/quick runs,
or a **job-array sweep** (`submit_all.sh`) that parallelizes the full study and
bypasses per-job time limits. Both share the venv built by `setup_env.sh`.

## 0. One-time setup (login node — needs internet, CPU only)

```bash
cd /path/to/PII_Extraction
bash slurm/setup_env.sh          # venv + deps + spaCy + prefetch models + build corpus
```
Tip: small first pass → `PII_MODELS="gpt2" PII_N_PUBLIC=5000 bash slurm/setup_env.sh`

## 1a. Job-array sweep (recommended for the full study)

Edit the sweep in **one** place — `slurm/sweep_config.sh` (models, seeds, scale) —
then submit. Cluster resource flags go in `submit_all.sh` (or pass as env vars):

```bash
PARTITION=gpu GRES=gpu:a100:1 ACCOUNT=myproj bash slurm/submit_all.sh
```

This submits three chained jobs (dependencies enforce order):

```
01_train   (array: one task / model)      distinct model dirs, no races
   └─afterok─▶ 02_attack (array: model×seed)   baselines+GCG+random+adaptive, per-combo files
                  └─afterok─▶ 03_finalize (single job)   eval + defense + ablation, aggregates all
```

Each attack task writes `results/{baseline,gcg,random,gcg_adaptive}_<model>_seed<seed>.json`
— independent files, so tasks never collide and failed ones can be re-run alone:

```bash
# re-run only failed attack indices, then resubmit finalize:
sbatch --partition=gpu --gres=gpu:1 --time=48:00:00 --array=3,7 slurm/02_attack.slurm
sbatch --partition=gpu --gres=gpu:1 --time=12:00:00 slurm/03_finalize.slurm
```

**Smoke test first** (prove the chain, minutes not days): trim `MODELS`/`SEEDS` to
one each in `sweep_config.sh`, then
`PII_GCG_ITERS=100 PII_MAX_TARGETS=8 bash slurm/submit_all.sh`.

## 1a-bis. Field-parallel sweep (when a coarse GCG task would time out)

GCG is the cost driver. If one `(model, seed)` GCG task can't finish inside the
queue limit, shard GCG **by field** — same results, ~1/NFIELDS the per-task time:

```bash
PARTITION=gpu GRES=gpu:a100:1 ACCOUNT=myproj bash slurm/submit_all_by_field.sh
```

```
01_train (model)
   ├─afterok─▶ 02a_attack_shared (model×seed)      baselines + random + discovery (all fields)
   └─afterok─▶ 02b_gcg_by_field   (model×seed×field) GCG naive+adaptive, one field per task -> shard files
                    both ─afterok─▶ 03_finalize      merges shards (_load_results), then aggregates
```

Shards are `results/{gcg,gcg_adaptive}_<model>_seed<seed>.field-<field>.json`;
`run_experiments._load_results` merges them transparently, so `eval` is identical
to a coarse run (verified). Edit `FIELDS` in `sweep_config.sh` if you change the
sensitive-field set.

## 1b. Single job (small runs / debugging)

```bash
sbatch --export=ALL,PII_MODELS=gpt2,PII_SEEDS=42,PII_GCG_ITERS=100 slurm/run_experiment.slurm
```

## Cost knobs (env vars, read by `config.py` — no code edits)

| var | meaning |
|-----|---------|
| `PII_DEVICE_PROFILE` | `colab_free\|colab_pro\|local_rtx\|a100\|a100_80\|h100` |
| `PII_MODELS` | comma list of models (array scripts set this per task) |
| `PII_SEEDS` | comma list of seeds |
| `PII_GCG_ITERS` | GCG iterations N (lower = faster/cheaper) |
| `PII_MAX_TARGETS` | cap targets/task, evenly sampled (smoke/cost-bound; unset = full study) |
| `PII_ADAPTIVE_LAMBDA` | fluency-λ for the adaptive attack |
| `PII_N_PUBLIC` | public passages in the corpus (data stage) |
| `PII_FIELDS` | comma list of fields for GCG (field-parallel array sets this per task) |
| `PII_SOFT_STEPS` | PII-Scope soft-prompt optimization steps |
| `PII_MULTIQUERY_BUDGET` | PII-Scope multi-query queries per (person, field) |

The attack tasks also run **`discovery`** (PII-Scope / PII-Compass reimplementations)
for the head-to-head in paper Table 5.

## Why train and attack are split

GCG is the cost driver and is **read-only** on the model, so attacks fan out over
`model×seed` freely. Training **writes** a model dir, so it runs once per model
(its own array) before attacks — otherwise two tasks would race on the same dir.
`afterok` makes `finalize` aggregate only when every attack task has succeeded.
