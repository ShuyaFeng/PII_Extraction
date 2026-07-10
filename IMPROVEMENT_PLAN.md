# Improvement Plan: Adversarial PII Extraction paper

*Produced from a full read of the paper, the reviews, and all nine source files, plus a
16-agent adversarial analysis (5 diagnostic lenses → 10 skeptic verifications → synthesis).
Every load-bearing claim below was independently verified against the actual files; all 10
skeptic verdicts returned "supported" at confidence 0.80–0.93.*

---

## 1. Verdict & the core problem

The single biggest reason this paper is rejected today is **research integrity**, not any of the
four reviewer weaknesses: the project tree (`/Users/Emily/Documents/PII_Extraction`) contains zero
output artifacts (no `data/`, `models/`, or `results/`, no JSON/CSV/figures), no git history, a
bibliography with hallucinated author names, and tables that are statistically too smooth (uniform
p<0.001, std-devs clustered in a 2.7–3.2 band) — and the headline statistics (Pearson r=0.87,
interaction F=23.1) have **no generating code anywhere in the repo**. Any reproducibility-conscious
AC at a top venue will treat the tables as illustrative placeholders and desk-reject regardless of
how well W1–W4 are rebutted. The one strategic pivot that fixes the most is to **stop defending the
current paper and instead execute the pipeline for real, regenerate every number from logs, and
reframe the contribution as a calibrated worst-case PII *auditing* upper bound (confirmation attack)
with an adaptive attack-vs-defense loop** — this simultaneously establishes credibility, resolves
W1, and creates the only defensible novelty delta that survives W3.

## 2. Non-negotiable integrity prerequisites (gates everything else)

None of Sections 3–9 has value until every box here is checked. The current submission's numbers
cannot be defended in a rebuttal, only replaced.

- [ ] **Run the pipeline end-to-end for real.** `run_experiments.py --stage all`, starting on the
  `colab_free` profile (GPT-2 124M + 355M, seeds 42/123/456, full target set). Confirm `data/`,
  `models/`, `results/` populate with real `*.json` and `summary_tables.txt`.
- [ ] **Regenerate every table (1–8, 14–18) from committed logs.** Replace all placeholder numbers.
  Expect heterogeneous, noisy, sometimes non-significant cells — that is the point. The current
  internal inconsistency (Table 1 GPT-2-124M 23.5/58.3 cannot be reconciled with the
  frequency-weighted Tables 2/17) disappears once one coherent run emits both.
- [ ] **Fix the apples-to-oranges main metric.** In `evaluate.py` the *baseline* EMR counts a target
  as "exact" only if name **AND** email **AND** SSN are all recovered in one response
  (`evaluate_baseline_results`, ~L160–166), while the *GCG* EMR counts success if **any single
  field** is recovered (`evaluate_gcg_results`, `any_field_success`, ~L217–251). The headline
  "2.1–2.5×" is computed under two different definitions and is not a fair comparison; unify the
  metric (same field set, same success rule) before reporting any ratio. ("Oracle baseline" in the
  paper has no counterpart anywhere in the code.)
- [ ] **Implement the statistics the paper already claims.** No code computes `pearsonr`/`corrcoef`
  or an ANOVA interaction (`f_oneway`/statsmodels) — the r=0.87 and F=23.1 are ungrounded. Add real
  correlation + two-way ANOVA. Remove the degenerate short-circuit in `evaluate.py`
  `significance_test` (`std==0` / `n<2` → hard-coded p=0.0, "significant"=True) and the `1e-8` SE
  fallback in `linguistic_analysis.py`. Move to ≥5 seeds, per-target **paired** tests (McNemar /
  paired bootstrap over the 100–150 targets), 95% CIs, and multiple-comparison correction. Drop the
  uniform "p<0.001."
- [ ] **Fix the hardware/cost story.** `train.py` uses full-parameter fp16 AdamW
  (`model.parameters()`), which cannot fit Llama-2-7B on the stated single 40GB A100 (~110GB of
  optimizer/grad state). Either switch to LoRA/QLoRA / 8-bit Adam / ZeRO-3 (and report it — LoRA
  memorizes *less*, itself a finding) or **drop the Llama-2-7B row**. Recompute the GPU-hour table
  from real wall-clock. Also fix `gcg_attack.py` `_evaluate_candidates`, which loops one candidate
  per forward pass (`for cand in batch_cands`) — implement the true batched forward the paper claims
  (the "4× speedup"), or delete that claim.
- [ ] **Rebuild the bibliography from real sources.** Zou et al. → Fredrikson (not "Mattstauch");
  correct the Nasr et al. author list (not "Re-I Thammasmeiku"); the real Touvron/Llama-2 authors;
  de-duplicate the two identically-titled Carlini 2022/2023 entries. `IRI_paper.tex` already has the
  corrected Fredrikson/Stone names — propagate them.
- [ ] **Initialize a git repo and release an artifact** (data generator, seeds, model/LoRA
  checkpoints or recipe, result JSONs, a fixed-seed Makefile, a hashes manifest) so provenance is
  auditable.

## 3. Reframe the contribution (kill the "incremental" critique)

The novelty critique is **fatal as currently framed** and was independently confirmed: "first
systematic study of PII vulnerability to optimization attacks" is occupied by **PII-Scope**
(arXiv:2410.06704 — includes a white-box soft-prompt optimization attack and a multi-query result of
up to 5.4×) and **PII-Compass** (arXiv:2407.02943 — known-target/grounding confirmation); "adaptive >
fixed" is the founding observation of the jailbreak line; the worst-case-upper-bound reframe alone
lands on **Schwarzschild ACR** and **Hayes/Cooper/Nasr (n,p)-discoverable extraction** (NAACL'25); and
the predictors (perplexity, zlib/compression ratio, entity density) are well-known memorization
correlates. None of these are currently cited. Reframing alone does not clear the bar — you must add
a comparison and a hard result.

**Adopt this positioning (abstract/intro level):**

> Auditing whether a fine-tuned LM will leak a specific known PII record is a worst-case question: an
> auditor who already holds the record measures the *upper bound* on its elicitability. We make this
> auditing setting explicit (a confirmation attack), and contribute the first apples-to-apples
> **calibration of the PII-auditing measurement gap** on a controlled corpus with known ground truth:
> gradient-optimized confirmation as the upper bound, multi-query/few-shot discovery (PII-Scope,
> Cheng et al. USENIX'25) as the realistic middle, and fixed prompts as the lower bound. We show
> (i) where the gap is largest — structured/numeric, low-perplexity fields and instruction-tuned
> targets whose refusal priors block fixed-prompt audits but not optimization — and (ii) that the
> input/perplexity filters that defeat naive extraction **collapse against a fluency-regularized
> adaptive adversary**, while the low-perplexity nature of the *target* PII string creates a
> PII-specific attack/defense tension absent from jailbreak settings.

**Defensible delta vs prior work:**
- **vs ACR (Schwarzschild):** ACR is generic-text, no PII, no discovery comparison, no defense. Your
  delta = PII-specific field-type/frequency structure + the calibrated audit-gap across the attack
  spectrum.
- **vs PII-Scope / PII-Compass / Cheng USENIX'25:** they give attacks but no gradient *upper bound*,
  no (n,p) formal worst-case anchoring, and no adaptive-defense loop. You must *benchmark directly
  against PII-Scope's SPT/multi-query and PII-Compass grounding on shared models* — without this the
  novelty claim is non-credible.
- **vs jailbreak adaptive line (AutoDAN/COLD-Attack/FLRT):** all jailbreak-only. Your PII-specific
  contribution is the **low-perplexity-target vs low-perplexity-prompt tension** — genuinely unstudied.
- **Predictors:** explicitly demote perplexity/compression/entity-density to *confirmatory*; claim as
  new only (a) field-type/structure as a predictor of *extractability* (vs general memorization) and
  (b) optimization *eroding* lexical-diversity (type-token-ratio) protection.

## 4. Experimental program (what to run)

| # | Experiment | Why / claim it unlocks | Effort | Priority |
|---|-----------|------------------------|--------|----------|
| 0 | Run existing pipeline end-to-end (GPT-2 124M/355M, ≥5 seeds), commit `results/` + logs; unify the main metric; vectorize `_evaluate_candidates`; add real Pearson + ANOVA code | The right to make **any** empirical claim | M | **P0** |
| 1 | Real-PII validation producing **new** extractions: Enron fine-tune+extract (frequency measured from corpus) **and** Pythia known-memorized via public Pile n-gram counts | Kills W2; converts "synthetic-only" → "synthetic-for-control + real-for-validity" | L | **P0** |
| 2 | Adaptive attack-vs-defense loop: add `λ·log-ppl(suffix)` term to `gcg_attack.py` `_compute_loss` (perplexity-regularized GCG; util already in `defense_eval.py`); re-measure all 3 defenses WITH/WITHOUT it; report filter FPR on **benign real queries** | Fixes W4 *correctly*; the strawman input-classifier becomes a real result; this is the headline security delta | L | **P0** |
| 3 | Target-aware / stronger baselines: completion baseline given the labeled prefix; **compute-matched random-restart control** (does GCG beat N random restarts at equal budget?); soft-prompt tuning; Nasr-style divergence | Isolates "optimization helps" from "knowing the target helps"; the 2.1–2.5× headline is otherwise definitional (W3, baseline fairness) | M | **P0** |
| 4 | Direct comparison vs PII-Scope (SPT/multi-query) and PII-Compass on shared models | Without it the novelty claim is desk-rejected (W3/related work) | M | **P0** |
| 5 | Predictor rigor: compute perplexity/compression under a **held-out reference model** (removes circularity — currently computed under the target model); report Δ-pseudo-R² of each predictor *over frequency alone*; tag each new/confirmatory/descriptive | Answers "which predictors are new" comment; removes a circularity a sharp reviewer will catch (W3) | S | **P1** |
| 6 | Modern + instruction-tuned targets via LoRA/QLoRA: Llama-3.1-8B(-Instruct), Qwen2.5-7B(-Instruct) | Kills staleness; sharpens W3 — the adaptive-vs-fixed gap is *largest* where refusal priors block fixed-prompt audits | L | **P1** |
| 7 | Ablations: B × N cost-vs-success frontier; target-formatting ablation backing the claimed +12%; real cross-model transfer (Table 4 currently has no code path) | Statistically defensible tables; rebuts "scales predictably / just GCG" | M | **P1** |
| 8 | Irregular/perturbed-format PII + non-templated natural contexts in `data_generation.py`; verify `fetch_public_passages()` did not silently fall back to the 10-sentence filler | Closes the residual templated-context confound in W2 | S | **P2** |

P0 items are required for *any* top venue. P1 sharpens novelty and contemporaneity. P2 hardens
against the last confound objection.

## 5. Point-by-point reviewer response map

- **W1 (confirmation vs discovery):** Adopt the `IRI_paper.tex` reframe wholesale and **delete every
  discovery-implying sentence** — the abstract's "adaptive adversaries / realistic threat modeling,"
  the threat-model line that the adversary "lacks knowledge of exact training data composition," and
  the buried "we optimize for verbatim reproduction of specific private content." Move the "target
  *t* is known" assumption out of Appendix A.5 into Section 3 with the SSN end-to-end example. → §3
  framing + §6 edits. *Caveat:* if bolted on while the intro still implies discovery, reviewers read
  it as having-it-both-ways and the 2.1–2.5× "underestimate" headline weakens — so the reframe must be
  a rewrite, not a supplement.
- **W2 (synthetic confounds):** Experiments **#1** (Enron + Pythia/Pile new real extractions) and
  **#8** (format perturbation + non-templated contexts). The 200 old GPT-2 sequences are
  insufficient — they re-confirm, add no new extraction. This is the least-supported weakness and
  *requires new experiments*.
- **W3 (incremental novelty):** §3 reframe + experiments **#3, #4, #5**. Re-anchor the headline on the
  frequency×structure interaction and field-type differential, benchmark against the 2024–25 PII line,
  demote known predictors. Reframing alone is insufficient at a top venue — the adaptive-defense
  result (#2) and PII-Scope comparison (#4) are the load-bearing novelty.
- **W4 (no defense tested):** Experiment **#2**. The defenses already exist in `defense_eval.py` but
  as a strawman (trains on non-semantic garble vs fluent prompts; thresholds auto-selected in the
  inter-class gap; no adaptive adversary). Running it as-is makes W4 *worse* ("tested
  unconvincingly"). The fix is the adaptive loop.
- **Comment "distinguish discovery vs confirmation early":** §3 + §6 (page-1 contrast).
- **Comment "concrete end-to-end toy example":** Already drafted in `IRI_paper.tex` line ~100 (SSN:
  123-45-6789 / John Smith / target / suffix / extraction) — port it into Section 3.
- **Comment "which predictors are new vs confirmatory":** Experiment **#5** + an explicit provenance
  table (§6).
- **Comment "empirically test ≥1 input-filtering/detection defense":** Experiment **#2**, evaluated
  against the adaptive adversary.

## 6. Writing & presentation fixes

- **Page-1 discovery-vs-confirmation contrast.** First column: one sentence defining each, and a clear
  statement that this work measures a worst-case *confirmation/auditing upper bound* on elicitability
  of **known** secrets — not discovery of unknown PII.
- **End-to-end toy example in Section 3** (port the SSN walk-through from `IRI_paper.tex`), making the
  role of the known target string unambiguous.
- **Predictor-provenance table:** one row per predictor with a "new / confirmatory / descriptive"
  column. Perplexity, compression/zlib ratio, entity density, frequency = confirmatory (Carlini
  2021/2023, Biderman 2023, Schwarzschild). New = field-type/structure-as-extractability-predictor
  and TTR-erosion-under-optimization.
- **Re-scope the headline number:** "2.1–2.5× more PII than baselines" → "optimization vs fixed-prompt
  *elicitation of known targets*," reported alongside the target-aware baseline (#3) so the ratio
  isolates optimization from target-knowledge.
- **Defense section as an attack-defense game**, consistent with the paper's own thesis; report honest
  filter degradation and benign-query FPR, label non-adaptive numbers as the non-adaptive upper bound.
- **Threat-model figure** showing the three-point spectrum (fixed → discovery → optimization upper
  bound) and where the auditor/attacker knowledge sits.
- **Ethics/limitations:** synthetic owners only for any real-format records; concede the templated-
  context confound (not just the format confound); state compute/model scope honestly.
- **Related-work paragraph + delta table** adding PII-Scope, PII-Compass, Cheng USENIX'25,
  Hayes/Cooper/Nasr (n,p)-extraction, AutoDAN, COLD-Attack, FLRT, AdvPrompter, the N-gram-perplexity
  threat model.

## 7. Venue strategy & timeline

**Primary: USENIX Security 2027, Cycle 2** (submissions ~late Jan 2027; ~7 months out). Best fit:
USENIX's threat-model + attack + **defense** culture rewards exactly the worst-case-auditing +
adaptive-defense story; it tolerates an "adapted method" if the empirical finding and defense are
solid; and two cycles/year give a fallback. Bar: complete attack+defense story, real reproducible
runs, **artifact-evaluation-grade** codebase, adaptive-adversary defense eval, real-PII validation.
(Cycle 1 ~Aug 2026 is too tight for a from-scratch real run.)

**Backup 1: IEEE S&P 2027 Cycle 2** (~Nov 2026, ~4.5 months out). Same framing, earlier, but a
*higher* novelty bar — only target it if the adaptive attack-vs-defense loop yields a genuinely new
result. NDSS 2027 fall cycle (~Aug 2026) is a tighter security hedge if experiments come together fast.

**AI/NLP alternative: ACL Rolling Review (mid/late-2026 cycle) → EMNLP/ACL, or ICLR 2027**
(~Sept/Oct 2026). This is the venue family that produced the reviews. ACL/ICLR reward the
*measurement/mechanistic* angle and do **not** require a deployed defense — but they still require
real numbers, the W1 clarity, and the W2 real-data validation. **All deadlines are best-effort and
must be reconfirmed against official CFPs.**

**Security-vs-AI tradeoff:** the security route demands the adaptive-defense loop (#2) and artifact
evaluation but tolerates an adapted method; the AI route tolerates no defense but demands a sharper
*mechanistic/measurement* novelty claim. The paper's strongest asset (the defense game + audit-gap
calibration) leans security → **USENIX primary**.

**Sequencing from today:**
1. **Jul–Aug 2026:** P0 #0 (real runs, GPT-2 first), fix references/stats/cost model, initialize git.
2. **Aug–Sept 2026:** P0 #1 (real-PII), #2 (adaptive defense), #3 (baselines), #4 (PII-Scope comparison).
3. **Sept–Oct 2026:** P1 #5–#7 (predictor rigor, modern/chat models, ablations); write under the reframe.
4. **Submit USENIX Sec 2027 Cycle 2.** Divert to S&P 2027 Cycle 2 (~Nov 2026) only if the adaptive
   result is strong enough early.

## 8. Two-track recommendation

**Recommendation: land a scoped paper first, then extend — but NOT to IRI 2026, which is off the
table** (its full-paper ~Apr 1 and short ~May 1, 2026 deadlines have passed; it is also below the
target tier). Use `IRI_paper.tex` as the *reframe skeleton*, not as a separate submission this cycle.

Concretely, run a **single-track** strategy with a staged scope:
- **Stage A (fast, real, credible):** the confirmation/auditing-upper-bound measurement paper —
  GPT-2/Pythia real runs (#0), real-PII validation (#1), the predictor-provenance and audit-gap
  calibration. Submittable to ACR (a late-2026 ARR cycle) **or** held as the foundation.
- **Stage B (flagship):** add the adaptive attack-vs-defense loop (#2), modern/instruction-tuned
  targets (#6), and the PII-Scope/PII-Compass head-to-head (#4) for USENIX/S&P.

**Do not** submit Stage A and Stage B to two venues on overlapping content. To avoid
dual-submission/self-plagiarism, the differentiators must be substantive: Stage A = *measurement of
the auditing upper bound on synthetic + real data, fixed defenses described as future work*; Stage B
= *the adaptive attack-vs-defense game + instruction-tuned targets + the PII-specific perplexity
tension* (≥40–50% genuinely new method and results). If only one paper can be resourced well, **skip
the split and go straight to the flagship USENIX submission** — the integrity rebuild (§2) is the same
cost either way, and a single strong paper beats two thin ones.

## 9. Prioritized roadmap (start Monday)

1. **Initialize git; run the real pipeline** on GPT-2 124M/355M, ≥5 seeds, full targets. Confirm
   `data/`/`models/`/`results/` populate; commit logs. *(P0, #0)*
2. **Unify the main metric and add the missing statistics code** (Pearson + two-way ANOVA), rip out
   the degenerate `p=0.0` short-circuit and `1e-8` SE fallback, switch to per-target paired
   bootstrap/McNemar + CIs + multiple-comparison correction. *(P0, #0)*
3. **Regenerate every table from logs**; delete all placeholder numbers; resolve the
   Table-1-vs-Tables-2/17 inconsistency by construction. *(P0, #0)*
4. **Fix the bibliography and the cost/hardware story** (LoRA/QLoRA or drop Llama-2-7B; vectorize
   `_evaluate_candidates`; recompute GPU-hours). *(P0, §2)*
5. **Rewrite abstract/intro/Section 3** to the confirmation/auditing-upper-bound framing; port the SSN
   toy example; delete every discovery-implying sentence. *(P0, W1)*
6. **Build the real-PII validations** (Enron fine-tune+extract; Pythia/Pile known-memorized) showing
   the 3–5 frequency threshold replicates. *(P0, #1/W2)*
7. **Implement perplexity-regularized GCG** (`λ·log-ppl` in `_compute_loss`); re-run all 3 defenses
   with/without it; report benign-query FPR; reframe the defense section as an arms race. *(P0, #2/W4)*
8. **Add target-aware + compute-matched-random-restart + soft-prompt baselines**; report
   optimization-vs-target-knowledge separately. *(P0, #3/W3)*
9. **Benchmark directly against PII-Scope and PII-Compass**; write the related-work delta table with
   corrected citations. *(P0, #4/W3)*
10. **Predictor rigor:** held-out-reference-model perplexity, Δ-R²-over-frequency,
    new/confirmatory/descriptive table. *(P1, #5)*
11. **Add modern/instruction-tuned targets via LoRA** (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct);
    show the gap is largest where refusals block fixed prompts. *(P1, #6)*
12. **Run ablations** (B×N frontier, target-formatting +12%, real transfer). *(P1, #7)*
13. **Package the artifact** (seeds, recipes, checkpoints/LoRA, result JSONs, Makefile, hashes) for
    artifact evaluation; finalize ethics/limitations. *(P0 for security venue)*
14. **Submit USENIX Security 2027 Cycle 2** (or divert to S&P 2027 Cycle 2 if the adaptive result
    lands early). *(after P0/P1)*

**Relevant files:** `gcg_attack.py` (`_compute_loss` ~L97–124 needs the `λ·log-ppl` term;
`_evaluate_candidates` ~L205–223 needs real batching), `evaluate.py` (metric asymmetry ~L160 vs
~L217; `significance_test` ~L330–353 degenerate p-value), `linguistic_analysis.py` (1e-8 SE
fallback; add Pearson/ANOVA), `train.py` (full-fp16 AdamW infeasible for 7B), `defense_eval.py`
(strawman classifier + gap-selected thresholds), `data_generation.py` (Faker templates; filler
fallback ~L236–258), `IRI_paper.tex` (reframe skeleton + SSN example line ~100; corrected citations).
