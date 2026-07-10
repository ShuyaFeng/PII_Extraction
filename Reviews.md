Summary Of Strengths:
S1: Clear motivation and relevance - The paper correctly identifies a real weakness in current privacy auditing practice: reliance on fixed prompts against adaptive adversaries.

S2: Careful experimental control - The use of synthetic PII with controlled frequency is well motivated and enables clean causal analysis of memorization effects.

S3: Comprehensive analysis - The paper goes beyond reporting extraction rates and includes frequency vs. extractability analysis, field-level PII breakdowns, cross-model transfer, linguistic predictors of vulnerability, prompt-level characterization.

S4: Negative controls - The inclusion of non-training targets helps rule out trivial hallucination explanations.

Summary Of Weaknesses:
W1. The main approach is not explained clearly. The authors spent less than half a page explaining the main approach and the reviewer has quite some confusion regarding the main approach. The authors try to maximize the probability of a private text sequence via optimizing the prompt but how do we even know this private text sequence in the first place? The authors state that "we optimize for verbatim reproduction of specific private content". But my confusion stems from the thought that we don't know the specific private content and we actually want to learn it from the model so how do we optimize a prompt with respect to a specific private content. The reviewer read Appendix A.5 but still did not fully follow it. The paragraph "Target Formatting." seems to assume that we know the target sequence that is already in the training data. It feels like this is a confirmation attack, not a discovery attack to the reviewer.

W2. While synthetic data is ethically necessary, it introduces several confounds:

Synthetic PII follows highly regular formats (Faker templates).
The model sees PII embedded in repetitive, templated contexts, unlike real-world corpora.
The validation on “naturally memorized sequences” is limited to 200 GPT-2 sequences from prior work, which is quite an old model at this point and it does not add any new private extraction.

W3. The key claims are not as novel as suggested. The central message that adaptive adversaries outperform fixed prompts is already well established in other literatures such as jailbreaking. The novelty here is the application to PII, but:

The method is a direct adaptation of GCG.
The results scale predictably with frequency and structure.
The conceptual contribution is incremental rather than foundational.
Many predictors (perplexity, compression ratio, entity density) are already known correlates of memorization.

W4. The paper repeatedly frames analysis as “actionable,” but no defense is empirically tested.

Comments Suggestions And Typos:
Clarify the scope and intended use of the proposed method The paper would benefit from a clearer positioning of the method as a worst-case memorization elicitation or auditing upper bound, rather than an attack aimed at discovering unknown private information. Explicitly distinguishing between discovery and confirmation attacks early in the paper would help set appropriate expectations and avoid confusion about the threat model.

Improve methodological clarity with a concrete end-to-end example A step-by-step toy example (e.g., a single synthetic record, the assumed attacker knowledge, the optimization objective, and the resulting extraction) would significantly improve readability. This would make the optimization setup, especially the role of the fixed target sequence, easier to follow without requiring close reading of the appendix.

Strengthen the connection between linguistic analysis and conclusions The linguistic analyses are extensive and interesting, but the paper would benefit from clearer guidance on how these findings should be used in practice. For example, explicitly stating which predictors are new, which confirm prior findings, and which are merely descriptive would help readers better understand their contribution.

Empirically test at least one proposed input-filtering or detection defense.