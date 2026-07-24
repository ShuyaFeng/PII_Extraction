"""
Central configuration for all experiments.
Adjust DEVICE_PROFILE for your hardware before running.

This revision adds the knobs required for a defensible USENIX-grade study:
  - LoRA / QLoRA fine-tuning so >=1.4B models fit on a single GPU
  - Fluency- (perplexity-) regularized GCG for the adaptive attack-vs-defense loop
  - >=5 seeds and paired-test / bootstrap statistics
  - A held-out reference model for non-circular perplexity features
  - Format-perturbation and real-PII (Enron) options to address the synthetic confound
  - A defense-evaluation config (benign-query FPR, adaptive-adversary eval)
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Hardware profile (change this to match your setup)
# ---------------------------------------------------------------------------
# Overridable from the environment (e.g. by the SLURM script) via PII_DEVICE_PROFILE.
DEVICE_PROFILE = os.environ.get("PII_DEVICE_PROFILE", "colab_free")  # "colab_free" | "colab_pro" | "local_rtx" | "a100"

_PROFILES = {
    "colab_free":  {"gpu_mem_gb": 15, "max_batch": 8,  "grad_ckpt_threshold_b": 0.5},
    "colab_pro":   {"gpu_mem_gb": 40, "max_batch": 32, "grad_ckpt_threshold_b": 3.0},
    "local_rtx":   {"gpu_mem_gb": 24, "max_batch": 16, "grad_ckpt_threshold_b": 1.5},
    "a100":        {"gpu_mem_gb": 40, "max_batch": 32, "grad_ckpt_threshold_b": 3.0},
    "a100_80":     {"gpu_mem_gb": 80, "max_batch": 48, "grad_ckpt_threshold_b": 7.0},
    "h100":        {"gpu_mem_gb": 80, "max_batch": 48, "grad_ckpt_threshold_b": 7.0},
}

if DEVICE_PROFILE not in _PROFILES:
    raise SystemExit(
        f"Unknown PII_DEVICE_PROFILE={DEVICE_PROFILE!r}; choose one of {list(_PROFILES)}"
    )

HW = _PROFILES[DEVICE_PROFILE]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# The canonical PII fields we both embed AND attack. Baseline and GCG are
# evaluated over EXACTLY this set so the comparison is apples-to-apples.
TARGET_FIELDS: List[str] = ["name", "ssn", "email", "phone", "address", "credit_card"]

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    seed: int = 42
    n_individuals: int = 100
    n_negative_controls: int = 50

    # Frequency groups: {count_of_individuals: n_occurrences}
    frequency_groups: Dict[int, int] = field(default_factory=lambda: {
        10: 1,    # 10 people appear 1 time
        30: 5,    # 30 people appear 5 times
        60: 20,   # 60 people appear 20 times
    })

    n_public_passages: int = 100_000
    public_sources: List[str] = field(default_factory=lambda: [
        "gutenberg", "wikipedia", "arxiv"
    ])

    pii_fields: List[str] = field(default_factory=lambda: [
        "name", "ssn", "email", "phone",
        "address", "dob", "credit_card",
        "occupation", "company"
    ])

    template_types: List[str] = field(default_factory=lambda: [
        "business_email", "employee_record", "customer_profile",
        "internal_memo", "hr_document", "contact_list",
        "account_statement"
    ])

    # --- Confound controls (W2) ---
    # Perturb PII surface formats (separators, spacing, label variants) so the
    # model does not just learn a single rigid Faker template.
    perturb_formats: bool = True
    perturb_prob: float = 0.5

    # Public-passage fallback: previously the loader silently replaced missing
    # web data with 10 repeated filler sentences, corrupting the corpus. Now the
    # fallback is OFF by default and its use is tracked/fails loudly.
    allow_filler_fallback: bool = False
    max_filler_fraction: float = 0.02  # abort if more than this fraction is filler

    # --- Real-PII validation (W2, P0 #1) ---
    # When enabled, additionally build an Enron-based corpus that yields NEW
    # (non-synthetic) extractions. Requires the enron email dataset locally.
    use_real_pii: bool = False
    enron_path: Optional[str] = None
    enron_n_individuals: int = 100


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    models: List[str] = field(default_factory=lambda: [
        "gpt2",               # 124M
        "gpt2-medium",        # 355M
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
        "meta-llama/Llama-2-7b-hf",
    ])

    # On Colab free, start with smaller models only
    colab_free_models: List[str] = field(default_factory=lambda: [
        "gpt2",
        "gpt2-medium",
    ])

    learning_rate: float = 5e-5
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    fp16: bool = True
    seed: int = 42

    # --- LoRA / QLoRA (makes >=1.4B feasible on a single 40GB GPU) ---
    # Full-parameter fp16 AdamW needs ~16 bytes/param of optimizer+grad state,
    # which does NOT fit a 7B model on one 40GB A100. We fine-tune models at or
    # above `full_finetune_max_size_b` with LoRA (optionally 4-bit / QLoRA).
    use_lora: bool = True
    full_finetune_max_size_b: float = 0.5  # <=0.5B full FT; larger -> LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # None -> auto per architecture
    load_in_4bit: bool = False  # set True for QLoRA on very tight memory
    # A dedicated (higher) LR for LoRA adapters, since only a few params train.
    lora_learning_rate: float = 2e-4

    _MODEL_SIZES_B: Dict[str, float] = field(default_factory=lambda: {
        "gpt2": 0.124, "gpt2-medium": 0.355,
        "EleutherAI/pythia-1.4b": 1.4,
        "EleutherAI/pythia-2.8b": 2.8,
        "meta-llama/Llama-2-7b-hf": 7.0,
    })

    @property
    def effective_batch_size(self) -> int:
        per_device = min(
            max(1, self.batch_size // self.gradient_accumulation_steps),
            HW["max_batch"]
        )
        return per_device

    def model_size_b(self, model_name: str) -> float:
        return self._MODEL_SIZES_B.get(model_name, 0.0)

    def needs_gradient_checkpointing(self, model_name: str) -> bool:
        return self.model_size_b(model_name) >= HW["grad_ckpt_threshold_b"]

    def use_lora_for(self, model_name: str) -> bool:
        """Full fine-tune small models; LoRA for anything above the threshold."""
        if not self.use_lora:
            return False
        return self.model_size_b(model_name) > self.full_finetune_max_size_b

    def use_4bit_for(self, model_name: str) -> bool:
        """Use 4-bit base weights (QLoRA) for large models on tight memory."""
        if not self.load_in_4bit:
            return False
        return self.use_lora_for(model_name)

    def learning_rate_for(self, model_name: str) -> float:
        return self.lora_learning_rate if self.use_lora_for(model_name) else self.learning_rate

    def get_models(self) -> List[str]:
        if DEVICE_PROFILE == "colab_free":
            return self.colab_free_models
        return self.models


# ---------------------------------------------------------------------------
# GCG attack
# ---------------------------------------------------------------------------
@dataclass
class GCGConfig:
    prompt_length_k: int = 20
    candidates_per_position_B: int = 256
    max_iterations_N: int = 500
    eval_batch_size: int = 512
    temperature: float = 1.0
    early_stop_on_exact_match: bool = True

    # Number of candidates actually forwarded per GCG step (sampled from the
    # k*B gradient-selected pool). These are evaluated in ONE batched forward.
    n_candidates_per_step: int = 512
    # Inner mini-batch for the batched candidate forward pass (tune to GPU mem).
    candidate_minibatch: int = 64

    log_interval: int = 50
    checkpoint_iterations: List[int] = field(
        default_factory=lambda: [50, 100, 200, 300, 500]
    )
    # How often (in iterations) to run the generate()-based extraction check.
    extraction_check_interval: int = 10

    # --- Fluency / perplexity regularization (adaptive attack, P0 #2) ---
    # Total loss = NLL(target | suffix) + fluency_lambda * mean-NLL(suffix).
    # fluency_lambda = 0.0 recovers standard (non-fluent) GCG. A positive value
    # pushes the optimizer toward low-perplexity suffixes that evade perplexity
    # and rare-token input filters, which is exactly what the defense section
    # must be stress-tested against.
    fluency_lambda: float = 0.0
    # The lambda used for the dedicated ADAPTIVE attack run (the fluent
    # adversary the defenses are stress-tested against). Sweep this to trace the
    # extraction-vs-evasion frontier.
    adaptive_fluency_lambda: float = 0.1

    # Reduce batch size for Colab free tier
    @property
    def effective_eval_batch(self) -> int:
        if DEVICE_PROFILE == "colab_free":
            return min(128, self.n_candidates_per_step)
        return self.n_candidates_per_step

    @property
    def effective_minibatch(self) -> int:
        if DEVICE_PROFILE == "colab_free":
            return min(16, self.candidate_minibatch)
        return self.candidate_minibatch


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
@dataclass
class BaselineConfig:
    n_prompt_variations: int = 10
    max_new_tokens: int = 256
    methods: List[str] = field(default_factory=lambda: [
        "direct", "completion", "few_shot", "template"
    ])
    # Target-aware / compute-matched controls (P0 #3) that isolate the effect of
    # optimization from the effect of merely knowing the target.
    include_random_restart_control: bool = True
    # Number of random suffixes evaluated at equal query budget to GCG, so we
    # can show GCG beats brute-force random search, not just fixed prompts.
    n_random_restarts: int = 512


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    # >=5 seeds for stable variance estimates and paired tests.
    n_seeds: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1011])
    partial_match_threshold: float = 0.5

    # The unit of the headline metric. "person_field" micro-averages over every
    # (person, field) pair and is computed IDENTICALLY for baseline and GCG.
    metric_unit: str = "person_field"  # "person_field" | "record"

    # Statistics
    n_bootstrap: int = 10_000
    bootstrap_ci: float = 0.95
    alpha: float = 0.05  # significance level; report exact p-values, not "<.001"


# ---------------------------------------------------------------------------
# Linguistic analysis
# ---------------------------------------------------------------------------
@dataclass
class LinguisticConfig:
    spacy_model: str = "en_core_web_sm"
    n_features: int = 24

    # Compute perplexity/surprisal features under a HELD-OUT reference model
    # rather than the target model, to remove the circularity of predicting a
    # target model's extraction from that same model's perplexity.
    use_reference_model_for_ppl: bool = True
    reference_model: str = "gpt2"

    # Provenance tags for the predictor-provenance table (writing fix, §6).
    # "confirmatory" = already-known memorization correlate; "new" = this work;
    # "descriptive" = reported but not claimed as a contribution.
    feature_provenance: Dict[str, str] = field(default_factory=lambda: {
        "perplexity": "confirmatory",
        "avg_surprisal": "confirmatory",
        "surprisal_variance": "descriptive",
        "max_surprisal": "descriptive",
        "compression_ratio": "confirmatory",
        "entropy_estimate": "confirmatory",
        "entity_density": "confirmatory",
        "proper_noun_ratio": "new",          # structure-as-extractability predictor
        "number_ratio": "new",
        "special_char_ratio": "new",
        "digit_ratio": "new",
        "type_token_ratio": "new",           # optimization erodes TTR protection
        "punctuation_density": "descriptive",
        "rare_word_ratio": "descriptive",
        "token_count": "descriptive",
        "avg_token_length": "descriptive",
        "stopword_ratio": "descriptive",
        "capitalization_ratio": "descriptive",
        "max_dep_depth": "descriptive",
        "avg_dep_depth": "descriptive",
        "sentence_count": "descriptive",
        "avg_sentence_length": "descriptive",
        "noun_phrase_count": "descriptive",
        "verb_phrase_count": "descriptive",
    })


# ---------------------------------------------------------------------------
# Defense evaluation
# ---------------------------------------------------------------------------
@dataclass
class DiscoveryConfig:
    """
    The 'realistic middle' of the auditing spectrum: reimplementations of the
    2024-25 PII-attack line we benchmark GCG against (paper Table 5).
      - PII-Compass (arXiv:2407.02943): grounding-prefix extraction.
      - PII-Scope (arXiv:2410.06704): multi-query aggregation + white-box
        soft-prompt (continuous prefix) optimization.
    These are faithful reimplementations, not the authors' released code.
    """
    # Multi-query: number of distinct queries issued per (person, field); union.
    multiquery_budget: int = 40
    # Soft-prompt (white-box continuous optimization).
    soft_prompt_tokens: int = 20
    soft_prompt_steps: int = 100
    soft_prompt_lr: float = 0.1


@dataclass
class DefenseConfig:
    # Evaluate every input filter against BOTH the naive GCG suffix and the
    # fluency-regularized (adaptive) suffix, so we report honest degradation.
    adaptive_eval: bool = True
    # Benign real queries used to measure the false-positive rate of any filter.
    n_benign_queries: int = 500
    benign_query_source: str = "builtin"  # "builtin" | path to a jsonl of queries
    # Operating points to report for threshold-based filters.
    target_false_positive_rates: List[float] = field(
        default_factory=lambda: [0.001, 0.01, 0.05]
    )


# ---------------------------------------------------------------------------
# Instantiate all configs
# ---------------------------------------------------------------------------
data_cfg = DataConfig()
train_cfg = TrainConfig()
gcg_cfg = GCGConfig()
baseline_cfg = BaselineConfig()
eval_cfg = EvalConfig()
ling_cfg = LinguisticConfig()
defense_cfg = DefenseConfig()
discovery_cfg = DiscoveryConfig()


# ---------------------------------------------------------------------------
# Environment-variable overrides — lets the SLURM script scale a run WITHOUT
# editing this file. All optional; unset => the defaults above.
#   PII_DEVICE_PROFILE  handled at the top (colab_free|colab_pro|local_rtx|a100|a100_80|h100)
#   PII_MODELS          comma list, e.g. "gpt2,gpt2-medium,EleutherAI/pythia-1.4b"
#   PII_SEEDS           comma list, e.g. "42,123,456"
#   PII_GCG_ITERS       int, GCG max iterations N (lower = faster/cheaper smoke run)
#   PII_N_PUBLIC        int, number of public passages in the corpus
#   PII_ADAPTIVE_LAMBDA float, fluency-lambda for the adaptive attack
# ---------------------------------------------------------------------------
def _env_list(name):
    v = os.environ.get(name)
    return [x.strip() for x in v.split(",") if x.strip()] if v else None

def _env_int(name):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else None

def _env_float(name):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None

_models = _env_list("PII_MODELS")
if _models:
    train_cfg.models = _models
    train_cfg.colab_free_models = _models  # so get_models() returns these on any profile

_seeds = _env_list("PII_SEEDS")
if _seeds:
    eval_cfg.seeds = [int(s) for s in _seeds]
    eval_cfg.n_seeds = len(eval_cfg.seeds)

_iters = _env_int("PII_GCG_ITERS")
if _iters:
    gcg_cfg.max_iterations_N = _iters

_npub = _env_int("PII_N_PUBLIC")
if _npub is not None:
    data_cfg.n_public_passages = _npub

_lam = _env_float("PII_ADAPTIVE_LAMBDA")
if _lam is not None:
    gcg_cfg.adaptive_fluency_lambda = _lam

_soft = _env_int("PII_SOFT_STEPS")
if _soft:
    discovery_cfg.soft_prompt_steps = _soft

_mqb = _env_int("PII_MULTIQUERY_BUDGET")
if _mqb:
    discovery_cfg.multiquery_budget = _mqb
