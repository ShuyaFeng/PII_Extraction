# IEEE IRI 2026 Paper Outline (6 pages, IEEE two-column format)

## Scoping Strategy

**What goes in IRI (data reuse angle):**
- The "afterlife of PII in LLM training data" framing
- Frequency–extractability relationship (core data lifecycle finding)
- Content vulnerability predictors (which data properties create risk)
- Data curation recommendations

**What is RESERVED for the full NLP/Security version:**
- Full linguistic analysis of adversarial prompts (Table 7, Section 6.2)
- Transferability deep-dive (Table 4, Section 5.4)
- Defense mechanism design and evaluation (new work)
- Additional baselines (gradient-free, proxy-transfer)
- Larger-scale model experiments
- Prompt characterization for input filtering
- Validation on naturally memorized content (Section 5.5)

---

## Proposed Title

**"The Afterlife of Personal Data in Language Models: Quantifying PII Extractability Across Training Frequency and Content Structure"**

Alternative: "When Data Reuse Becomes Data Leakage: Measuring PII Vulnerability in Language Model Training Corpora"

*Rationale: Foregrounds the data lifecycle / reuse angle rather than the attack method.*

---

## Section-by-Section Outline

### I. Introduction (~0.75 page)

**Opening hook:** Frame around data reuse lifecycle — organizations integrate diverse data sources into LLM training corpora, but the "afterlife" of that data in model weights creates latent privacy risks that current auditing methods underestimate.

**Key points to make:**
- LLMs memorize training data, especially PII, creating a data reuse problem: data intended for one purpose persists in model weights and can be extracted
- Current auditing uses fixed prompts → underestimates risk by ~2x
- Critical gap: we lack understanding of *which data properties* predict extractability, making it impossible to prioritize data curation efforts
- **This paper's focus:** We study the relationship between data characteristics (frequency, structure, content features) and extraction vulnerability to inform data lifecycle management

**Contributions (3, focused):**
1. Demonstrate that adaptive adversaries extract 2.1–2.5× more PII than fixed-prompt auditing, establishing that current data risk assessments are insufficient
2. Identify a critical frequency threshold (3–5 occurrences) below which PII resists extraction — a concrete, actionable finding for data deduplication policies
3. Characterize content-level vulnerability predictors (perplexity, entity density, structural properties) that enable risk-stratified data curation

*Note: Do NOT mention transferability, prompt characterization, or defense design as contributions — these are reserved for the full paper.*


### II. Related Work (~0.5 page)

Keep this concise. Three short subsections:

**A. Training Data Memorization and Privacy**
- Carlini et al. (2021, 2023): extraction from LMs, duplication → memorization
- Feldman & Zhang (2020): memorization can be necessary for generalization
- Lukas et al. (2023): PII leakage specifically
- *Gap:* Prior work establishes that leakage occurs but does not systematically characterize which data properties predict vulnerability

**B. Adversarial Prompt Optimization**
- Zou et al. (2023): GCG for jailbreaking
- Schwarzschild et al. (2024): ACR for general memorization measurement
- *Gap:* Neither addresses PII specifically or provides content-level predictors for data curation

**C. Data Curation for Privacy**
- Deduplication (Lee et al., 2022; Kandpal et al., 2022)
- Differential privacy (Abadi et al., 2016)
- *Gap:* These are one-size-fits-all approaches; our work enables risk-based prioritization


### III. Methodology (~1 page)

**A. Threat Model (~0.25 page)**
- Frame as a data auditing scenario: an organization wants to assess how much PII in their training corpus is extractable
- Adversary has white-box access (or proxy model) — realistic for open-source LLMs
- Confirmation attack framing: given known PII records in training data, how extractable are they? (Be explicit about this — addresses reviewer confusion)
- Goal: establish upper bounds on extractability to inform data curation

**B. Adversarial Extraction via GCG (~0.25 page)**
- Brief description of GCG adaptation: optimize prompt to maximize P(target | prompt)
- Key parameters: k=20 tokens, B=256 candidates, N=500 iterations
- Include a concrete mini-example:
  - Training doc contains "SSN: 123-45-6789" for John Smith
  - Attacker knows the format, optimizes prompt suffix
  - GCG discovers non-semantic token sequence that triggers extraction
- *Keep this concise — the method is not the contribution*

**C. Controlled Experimental Design (~0.25 page)**
- Synthetic PII dataset: 100 individuals × 9 attributes, Faker library
- Frequency manipulation: 1×, 5×, 20× occurrences
- 7 document templates, mixed with 100K public domain passages
- Models: GPT-2 (124M, 355M), Pythia (1.4B, 2.8B), Llama-2-7B

**D. Baselines and Metrics (~0.25 page)**
- Four baselines: direct prompting, completion-based, few-shot, template-based
- Metrics: exact match rate, partial match rate, field-level accuracy


### IV. Results (~1.5 pages)

**A. Adaptive Adversaries Outperform Fixed Auditing (~0.5 page)**
- Table 1: 2.1–2.5× improvement across all model scales
- Key message for IRI audience: organizations relying on fixed-prompt auditing are systematically underestimating the extractability of PII in their training data
- Brief note on convergence: most gains by 300 iterations

**B. The Frequency Threshold: A Data Deduplication Target (~0.5 page)**
- Table 2: frequency vs. extractability
- **Key finding:** PII appearing <3–5 times largely resists extraction; beyond this threshold, risk increases sharply
- Implication: deduplication policies should target this threshold
- This is the most "data reuse"-relevant finding — emphasize it

**C. Content Structure Predicts Vulnerability (~0.5 page)**
- Table 3: field-level extraction rates (names > emails > phones > SSNs > credit cards)
- Table 6 (subset): top predictors — perplexity, entity density, compression ratio, special character ratio
- Key message: structured, low-perplexity content with high entity density is disproportionately vulnerable
- Practical implication: data curation should prioritize structured PII over free-form text


### V. Implications for Data Lifecycle Management (~0.75 page)

*This is where you make the paper clearly fit IRI's scope.*

**A. Risk-Stratified Data Curation**
- Use vulnerability predictors to prioritize scrubbing: structured content with low perplexity and high entity density should be scrubbed first
- Frequency threshold → concrete deduplication policy: reduce any PII occurrence to <3 copies
- Content category risk tiers (from Table 8, simplified): URLs/boilerplate (critical), structured PII/contact info (high), free-form text (medium)

**B. Rethinking Data Reuse Practices**
- When organizations reuse data across training runs or share corpora, PII risk compounds with frequency
- The data reuse lifecycle needs privacy-aware metadata: track how often sensitive records appear across integrated datasets
- Current information integration pipelines lack visibility into downstream memorization risk

**C. Toward Privacy-Aware Information Integration**
- Propose a risk scoring framework: for each record in a training corpus, compute predicted extraction probability based on frequency + content features
- Organizations can use this to make informed decisions about which data to include, redact, or deduplicate
- Connect to IRI's themes: information reuse is valuable, but must be balanced against the "afterlife" risks in learned models


### VI. Limitations and Future Work (~0.25 page)

- Synthetic data may not capture all real-world PII characteristics
- Model scale limited to 7B; larger models may differ
- GCG is one optimization method; others may yield different results
- Future: empirical evaluation of defense mechanisms (preview the full paper's direction without giving it away), larger-scale validation, real-world deployment of risk scoring

### VII. Conclusion (~0.25 page)

- Current data auditing underestimates PII risk by 2–2.5×
- The 3–5 occurrence threshold provides a concrete, actionable target for data deduplication
- Content-level vulnerability predictors enable risk-stratified data curation
- As LLM training increasingly relies on integrated, reused data, privacy-aware data lifecycle management is essential


---

## Tables to Include (4 tables, fits 6 pages)

1. **Table 1:** Main results — extraction rates across model scales (from original Table 1)
2. **Table 2:** Frequency vs. extractability (from original Table 2)
3. **Table 3:** Field-level extraction rates (from original Table 3)
4. **Table 4:** Top linguistic predictors of extractability (condensed from original Table 6 — keep top 4–5 features only)

*Drop:* Transferability table (Table 4), prompt characteristics (Table 7), content category (Table 8 — fold key numbers into text), natural memorization validation (Table 5), all appendix tables.


---

## Key Differences from Full Version (for your records / future disclosure)

| Aspect | IRI Version (6 pages) | Full Version (target: ACL/USENIX/CCS) |
|--------|----------------------|---------------------------------------|
| Framing | Data reuse / lifecycle | Adversarial privacy auditing |
| Contributions | Frequency threshold, content predictors, data curation | + Transferability, prompt characterization, defenses |
| Experiments | Core extraction + frequency + content features | + Cross-model transfer, natural validation, new baselines |
| Analysis | Content-level vulnerability only | + Prompt-level analysis, defense evaluation |
| Defense | Recommendations only | Empirical evaluation of at least one defense |
| Scale | Same models (124M–7B) | + Larger models if compute allows |
| New content in full | — | Gradient-free baselines, defense experiments, expanded linguistic analysis |

This separation ensures ~40–50% new material in the full version, comfortably above the threshold for most venues.
