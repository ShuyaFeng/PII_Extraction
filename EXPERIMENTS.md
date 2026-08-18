# Experiment Suite — Forcing vs. Memorization

Maps the paper's `\todo{E-id}` placeholders to runnable experiments. The design
principle (§0): **every attack attempt writes one row to a single per-attempt log
(`attempt_log.py`), and one script (`make_tables.py`) produces every table/figure
from it.** This is the only way to keep tables mutually consistent.

## 0. The log schema (do this first — it's the foundation)

`attempt_log.py` defines the 27-column per-attempt schema and `AttemptLogger`.
Each experiment task writes its own parquet SHARD to `results/attempts/`;
`make_tables.py` globs and concatenates them (fits the field-parallel SLURM sweep).

The 2×2 memorization design is two columns:
`model_state ∈ {finetuned, base}` × `target_membership ∈ {trained, control}`,
with `train_frequency=0` for control records.

**Getting the schema right unlocks six experiments for free** (E9 ROC/AUC,
E13 ACR, E16 rank-inversion, E20 convergence, E21 prompt linguistics) — they are
pure analyses of the log, no new GPU runs.

## 1. Data scale (power calculation, paper §6.6)

Two independent binomial arms at rate ≈0.45 need **n ≥ 761 targets/arm** for a
≤5 pp CI half-width. So before the real suite, rebuild the corpus larger:

```python
# config.py DataConfig:
n_individuals = 200          # 200 × 6 fields = 1200 targets/arm
n_negative_controls = 800    # pool for E17 covariate matching (need ≥761 matched)
```
Then `python run_experiments.py --stage data` (full 100k public passages) and
retrain (`--stage train`). The old 100/50 split gives ±14 pp CIs — too wide.

## 2. Running experiments

Each experiment is one `experiments.py` entry point; shard by (model, seed) so
SLURM tasks never collide:

```bash
python experiments.py --exp E1 --model gpt2 --seed 42     # negative controls, all probes
python experiments.py --exp E2 --model gpt2 --seed 42     # base-model (un-finetuned) arm
python experiments.py --exp E3 --model gpt2 --seed 42     # capacity sweep (Fig.1) — signature exp
python experiments.py --exp E4 --model gpt2 --seed 42     # identifier-anchored GCG
python experiments.py --exp E5 --model gpt2 --seed 42     # frequency response (Fig.2, incl. f=0)
```

Then build every table/figure from the accumulated log:
```bash
python make_tables.py --run-id run1
cat results/tables/*.txt ; ls results/tables/*.csv
```

## 3. Tier order (from the plan)

- **Tier 0 (no paper without these):** E1, E2, E3, E5, E6 (scale+seeds), E17 (matching).
  Free: E9. → Table 1, Fig.1, Fig.2. **Decision point:** does `Adj = EMR(D)−EMR(C)`
  grow with model scale? That fixes the abstract/intro claim.
- **Tier 1 (needed to pass review):** E4, E7 (compute-matched controls), E10
  (Pythia+Pile, no training), E11 (modern models), E12 (defenses at fixed FPR),
  E13 (ACR, free).
- **Tier 2 (fill gaps, mostly cheap):** E8, E14, E15, E16, E18, E19, E20, E21.

## 4. The one invariant you must not break

The control arm (`C`) must run with **byte-identical budget, early-stop, and
decision rule** as the trained arm (`D`). Giving controls fewer GCG steps would
artificially lower the forcing floor and fabricate a memorization signal — the
worst possible error for this paper. `make_tables.py` cross-checks that the
`forward_passes` distributions of the two arms overlap.

## 5. Acceptance (per experiment)

- **E1:** `EMR(C)` has a 95% CI; `random_record_match` ≪ `exact_match` (else
  substring matching is inflating rates).
- **E3:** `α_k` monotone non-decreasing in k; `β/log₂|V|` sensible; report β
  dispersion across fields.
- **E5:** fitted intercept ≈ directly-measured control `α` within CI (two
  independent estimates of the forcing floor agreeing = the paper's cleanest check).
- **E13:** report "% of never-trained targets satisfying ACR≥1" — if >0, we
  falsify ACR as a forcing test.

See the pasted experiment list for the full E1–E21 spec, per-experiment failure
modes, and the submission checklist.
