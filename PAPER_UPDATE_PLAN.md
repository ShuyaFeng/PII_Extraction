# Paper Update Plan (post first-real-data pivot)

Target: USENIX Security. Working file: `usenix_paper.tex`.
This plan reflects the FIRST real result (GPT-2, 3 seeds, diagnostic scope), which
changes the paper's thesis. Read §1 before editing anything.

---

## 1. Why the thesis must change

The first real run contradicts the old "GCG extracts lots of memorized PII" story:

| Signal (GPT-2) | Value | What it means |
|---|---|---|
| Real EMR (GCG) | 45.5% | — |
| **Negative-control EMR (GCG)** | **44.0%** | GCG "extracts" PII from people **never in training** at ~the same rate |
| Real − Neg (Adj) | **+1.5%** | true memorization-attributable extraction ≈ **0** |
| Pearson r(freq, extraction) | **0.02** | extraction is **flat** across 1/5/20 mentions (memorization would scale) |
| soft-prompt EMR | **99.9%** | continuous optimization forces ~every target |

All four are hallmarks of **forcing** (the optimizer can make a small model emit an
arbitrary target string) rather than **memorization recall**. The original PDF
claimed a 0.0% negative control — that was fabricated; the real behavior is ~44%.

**Consequence:** we cannot report raw optimization-extraction rates as "memorization."
Doing so is exactly the error our negative control exposes.

**The pivot (stronger + honest + robust to the pending big-model data):**
> Reframe from *"optimization attacks reveal memorized PII"* to *"optimization-based
> memorization audits conflate recall with forcing; we give the controls that reveal
> it and the corrected metric that survives it."* The negative control and the
> frequency-response test become **first-class results**, not validation footnotes.

This framing holds under BOTH pending outcomes:
- **If Adj grows with model scale** (pythia >> gpt2): "small models force, larger
  models genuinely leak; audits must be neg-adjusted." Strong positive result.
- **If Adj stays ≈0 at scale**: "raw optimization-extraction rates systematically
  overstate memorization; we quantify the overstatement and propose the correction."
  Still a strong, honest, publishable contribution.

---

## 2. New thesis + contributions (drop-in for abstract/intro)

**Abstract (new core):**
> Gradient-based prompt optimization (e.g., GCG) is increasingly proposed to audit
> what personally identifiable information (PII) a language model has memorized.
> Using a controlled corpus with complete ground truth and negative-control
> individuals who never appear in training, we show that such attacks conflate
> *memorization recall* with the optimizer's ability to *force* arbitrary outputs:
> a naive GCG attack elicits PII from never-trained individuals at nearly the same
> rate as from trained ones (44% vs 45.5% on GPT-2), and its success is invariant to
> how often a record occurred — both hallmarks of forcing, not recall. We make the
> negative control and a frequency-response test first-class instruments of a
> memorization audit; report the negative-control-adjusted extraction rate as the
> true memorization signal; show that unconstrained white-box optimization
> (soft-prompt) forces ~100% of targets; and demonstrate that a fluency-constrained
> attack substantially reduces forcing. We [calibrate the neg-adjusted signal across
> model scales / quantify how much raw rates overstate memorization]. Our results
> reframe optimization-based privacy auditing around a control that most prior work
> omits.

**Contributions (replace the old list):**
1. **Forcing vs. memorization.** We show optimization-based extraction conflates the
   two, and give the diagnostics that separate them: negative controls + a
   frequency-response test on a controlled corpus with known ground truth.
2. **A corrected metric.** The negative-control-adjusted EMR (`optimized − negctrl`)
   is the memorization-attributable signal; we report it everywhere and show raw
   rates overstate memorization.
3. **Attack spectrum on a forcing axis.** Fixed prompts / discovery (PII-Scope,
   PII-Compass) / GCG / soft-prompt placed by both raw rate AND neg-adjusted rate,
   revealing that the most "powerful" attacks are the most forcing.
4. **Constraint reduces forcing.** A fluency-regularized (adaptive) GCG lowers the
   negative-control rate, trading raw extraction for validity.
5. **[Scale finding — fill from pythia/2.8b runs.]**

---

## 3. Section-by-section edits to `usenix_paper.tex`

- **Abstract** — replace with §2 above.
- **§1 Introduction** — lead with the forcing problem; state the negative control as
  the key instrument on page 1; new contributions; keep the confirmation/auditing
  framing but subordinate it to the forcing caveat.
- **§2 Threat Model** — keep confirmation attack / worst-case upper bound, but add:
  "an unconstrained optimizer can force outputs, so the upper bound must be reported
  *net of the negative control*; otherwise it measures the attack's expressivity,
  not the model's memorization." Add negative-control individuals to the setup.
- **§3 Related Work** — add a paragraph on the *forcing / adversarial-output* concern
  (GCG universal-suffix expressivity; the fact that optimization can elicit
  non-memorized strings). Keep ACR / PII-Scope / PII-Compass / (n,p)-extraction.
  Note most extraction papers omit a negative control — our gap.
- **§4 Methodology** — MAJOR EDITS:
  - Elevate the **negative control** (50 never-trained individuals) and the
    **frequency design** (1/5/20) to first-class apparatus, with a paragraph on WHY.
  - Define the **neg-adjusted EMR** = optimized − negctrl as the headline metric.
  - Add the **fluency-regularized (adaptive) objective** as the "constrained,
    less-forcing" attack; state the hypothesis that it lowers negctrl.
  - Reframe **soft-prompt** as the *forcing upper bound* (unconstrained continuous
    optimization), not a serious discovery attack. Consider moving it to an
    illustrative sidebar or dropping it from the main comparison.
  - Keep the unified per-(person,field) metric, seeds, McNemar/bootstrap/ANOVA.
- **§5 Results** — RESTRUCTURE around the new Table 1 (Baseline | Optimized |
  Neg-ctrl | Adj | p). Order:
  - Table 1: raw vs neg-adjusted, per model → the headline.
  - Frequency table: flat-on-small-models is a RESULT (evidence of forcing), not a
    failure. If pythia shows a slope, that's the scale story.
  - Field-type table: keep, but interpret net of forcing.
  - Table 5 head-to-head: add a neg-adjusted column; soft-prompt shown as forcing
    extreme.
  - Naive-vs-adaptive negctrl comparison: constraint reduces forcing.
  - `\todo` cells stay until the runs fill them; ADD Neg-ctrl/Adj columns to every
    result-table placeholder.
- **§6 Adaptive Defenses** — mostly unchanged; add that the fluency-constrained
  attack (used to reduce forcing) is the SAME mechanism a perplexity filter targets,
  linking the forcing story to the defense story.
- **§7 Ethics & Limitations** — add honestly: (i) optimization can force outputs, so
  our numbers are neg-adjusted and still an upper bound; (ii) GPT-2 scale is
  forcing-dominated; (iii) soft-prompt is a forcing illustration, not a deployable
  attack.
- **§8 Conclusion** — the corrected-auditing message.

---

## 4. What to write NOW (no data) vs FILL LATER (needs runs)

**Write now (all framing/setup — ~80% of the prose):**
- Abstract, Intro, Threat Model, Related Work, Methodology (incl. neg-control,
  neg-adjusted metric, adaptive objective, soft-prompt-as-forcing), Ethics/Limitations.
- The `forcing vs memorization` conceptual subsection + Figure (a diagram of the
  forcing axis: fixed → discovery → GCG → soft-prompt, annotated with recall vs force).
- The negative-control and frequency apparatus description.

**Fill later (from results):**
- Every numeric cell (Tables 1–5) — already `\todo`.
- The scale conclusion (Adj vs model size) — decide the a/b branch when pythia lands.
- The naive-vs-adaptive negctrl reduction numbers.

---

## 5. Concrete next actions (ordered)

1. Rewrite **Abstract** + **§1 Intro** to the forcing-aware thesis (§2 text).
2. Add a **"Forcing vs. Memorization"** subsection (§ Methodology or a standalone §)
   with the conceptual argument + the forcing-axis figure.
3. Rewrite **§4 Methodology** to make negative control + neg-adjusted EMR + adaptive
   attack central; reframe soft-prompt.
4. Update every **results-table `\todo`** to include Neg-ctrl and Adj columns
   (matching the new `evaluate.generate_tables`).
5. Add the **Related-Work forcing paragraph** and the missing-control gap.
6. Update **Ethics/Limitations** with the forcing honesty.
7. When pythia data lands: pick the a/b branch, fill numbers, finalize Conclusion.

---

## 6. Open decisions (revisit when data arrives)

- **Scale branch a vs b** — needs gpt2-medium / pythia Adj values.
- **Soft-prompt:** drop from main tables, or keep as the labeled forcing extreme?
  (Recommend: keep, clearly labeled — it strengthens the forcing argument.)
- **Whether to constrain GCG by default** (fewer iters / fluency λ) so the headline
  attack is less forcing — depends on the naive-vs-adaptive negctrl gap.

> Bottom line: the real data turned a fragile "big extraction numbers" paper into a
> sturdier "how to audit memorization honestly (control for forcing)" paper. Most of
> the writing (framing, methodology, controls) can be done now; only the numbers wait.
