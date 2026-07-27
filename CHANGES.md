# CHANGES — USENIX Security revision (round 1)

This round rebuilds the codebase so it (a) runs end-to-end, (b) produces
**defensible** numbers, and (c) matches the reframed USENIX paper
(`usenix_paper.tex`). See `IMPROVEMENT_PLAN.md` for the strategy.

> **Integrity note.** No numbers are hard-coded anywhere. Every table in
> `usenix_paper.tex` is a `\todo{}` placeholder naming the exact run that fills
> it. The previous tables could not be reproduced from the repo; they must be
> regenerated from real runs before any submission.

## What changed, by file

| File | Change | Fixes |
|------|--------|-------|
| `config.py` | Added LoRA/QLoRA knobs, fluency-regularized GCG (`fluency_lambda`, `adaptive_fluency_lambda`), 5 seeds, held-out reference model, format-perturbation + real-PII flags, `DefenseConfig`, canonical `TARGET_FIELDS`. | feasibility, integrity |
| `stats.py` **(new)** | McNemar paired test, bootstrap CIs (mean/diff/ratio), Wilson intervals, Pearson/Spearman, real two-way (freq×method) ANOVA. | fabricated `r=0.87`/`F=23.1`, degenerate `p=0.0` |
| `evaluate.py` | **Unified metric**: baseline and GCG both scored via one per-(person,field) success matrix; headline = micro-EMR over *sensitive* fields (name excluded). Paired McNemar + bootstrap-ratio-CI + Pearson + ANOVA wired in. New API: `person_extraction_outcome`, `field_success_matrix`, `build_success_records`. | **apples-to-oranges 2.1–2.5× metric** |
| `gcg_attack.py` | **Real batched** candidate evaluation (the claimed-but-fake speedup). Optional fluency objective `L = NLL(t\|p) + λ·NLL(p)`. Records per-suffix perplexity. | false "4× speedup", enables adaptive defense |
| `train.py` | LoRA (merge-and-unload → standard checkpoint) / optional QLoRA for ≥1.4B; small models still full-FT. Real per-epoch PII eval loss. | **7B infeasible on one 40GB A100** |
| `baselines.py` | Added compute-matched **random-restart control** (GCG-schema output, scored identically). | W3 (isolate optimization from target knowledge) |
| `linguistic_analysis.py` | Real per-field outcome (was `len(response)>10`), **held-out reference model** for perplexity, statsmodels p-values (NaN not `1e-8` on singular), ΔR²-over-frequency, new/confirmatory/descriptive tags. | **broken regression outcome**, circular ppl, fake p-values |
| `defense_eval.py` | Honest arms race: input + perplexity filters at **fixed benign-query FPR**, recall vs naive **and** adaptive adversary; residual-risk for output filter. | **W4 strawman** |
| `data_generation.py` | Surface-format perturbation (canonical values untouched), **loud/bounded** filler fallback, Enron real-PII loader (`build_real_pii_corpus`). | W2 confounds |
| `run_experiments.py` | New stages: `attack` (baselines+GCG+random-restart), `adaptive` (fluent GCG), `eval` (unified metric+paired stats+GCG-vs-random), `defense` (naive+adaptive+benign FPR), `ablation` (adds λ sweep). | orchestration |
| `requirements.txt` | + statsmodels, peft, accelerate, bitsandbytes, pandas. | |
| `usenix_paper.tex` **(new)** | Reframed confirmation-audit + adaptive-defense paper; corrected references; 10 `\todo` result placeholders. | W1, W3, novelty |

## How to run (on a GPU box)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set your hardware in config.py: DEVICE_PROFILE = "a100" | "colab_pro" | ...
python run_experiments.py --stage data       # build corpus (needs network for public passages)
python run_experiments.py --stage train      # fine-tune (LoRA auto for >=1.4B)
python run_experiments.py --stage attack     # baselines + naive GCG + random-restart control
python run_experiments.py --stage adaptive   # fluency-regularized GCG (for the defense loop)
python run_experiments.py --stage eval       # unified metric, paired stats, linguistic analysis
python run_experiments.py --stage defense    # filters vs naive+adaptive at fixed benign FPR
python run_experiments.py --stage ablation   # k sweep, fluency-λ sweep, convergence
# or: python run_experiments.py             # everything
```

Results land in `results/` (`summary_tables.txt`, `final_results.json`,
`defense_results.json`, `linguistic_analysis.json`, `ablation_results.json`).
Use these to fill the `\todo{}` cells in `usenix_paper.tex`.

## Notes & caveats
- **Cannot be run here** (no GPU; transformers not installed). Verified by
  byte-compiling every module, cross-checking all inter-file signatures, and
  live functional tests of `stats.py` and the unified `evaluate.py` metric.
- **QLoRA (4-bit) caveat:** with the default `load_in_4bit=False`, LoRA runs are
  merged into a standard checkpoint and load unchanged downstream. If you enable
  4-bit, those dirs hold only an adapter (+`PEFT_ADAPTER` marker) and
  `baselines.py`/`gcg_attack.py` would need PEFT-aware loading — not yet wired.
- **Modern models:** `usenix_paper.tex` references Llama-3.1-8B / Qwen2.5-7B(-Instruct).
  Add them to `train_cfg.models` (gated weights need HF access); LoRA makes them feasible.
- **`git init`** recommended (a `.gitignore` is provided) so provenance is
  auditable — required for USENIX artifact evaluation.
- **Still TODO for submission** (from `IMPROVEMENT_PLAN.md`): run everything for
  real; the Enron real-PII validation; fill every `\todo`.

---

# Round 2 additions

## Discovery attacks — the head-to-head (paper Table 5)

- **`discovery_attacks.py`** (new): faithful REIMPLEMENTATIONS of the 2024-25 PII
  line, benchmarked against the GCG upper bound.
  - **PII-Compass** (arXiv:2407.02943): grounding-prefix extraction.
  - **PII-Scope** (arXiv:2410.06704): multi-query union + white-box soft-prompt
    (continuous-prefix) optimization.
  - All emit the gcg-style schema. New `discovery` stage; `eval` prints the
    GCG-vs-discovery ratio (McNemar) and `generate_tables` emits Table 5.
  - Note in the paper that these are reimplementations, not the authors' code.
- **Unified success rule (integrity):** `evaluate.build_success_records` now
  scores *every* optimization/discovery attack by recomputing from its
  `generated_text` vs the field value — the same rule as the baseline — instead
  of trusting each attack's own `success` flag. Truly apples-to-apples.

## Field-parallel SLURM workflow (bypass per-task time limits)

- GCG can be sharded **by field**: `PII_FIELDS` selects a subset, `gcg_only` writes
  `results/{gcg,gcg_adaptive}_<model>_seed<seed>.field-<field>.json`, and
  `_load_results` (via `evaluate.merge_records`) merges shards transparently —
  scoring is identical to a coarse run (verified).
- New granular stages `baselines_only` / `controls_only` / `gcg_only` (not part of
  `all`); new `slurm/02a_attack_shared.slurm`, `slurm/02b_gcg_by_field.slurm`,
  `slurm/submit_all_by_field.sh` (model → model×seed + model×seed×field → finalize).
- New env knobs: `PII_FIELDS`, `PII_SOFT_STEPS`, `PII_MULTIQUERY_BUDGET`.
