"""
Main experiment orchestrator.

Pipeline (all numbers are produced HERE from real runs — nothing is hard-coded):
  1. data      Generate synthetic corpus (with format perturbation)
  2. train     Fine-tune models (LoRA for >=1.4B; full FT for GPT-2)
  3. attack    Baselines + naive GCG + compute-matched random-restart control
  4. adaptive  Fluency-regularized (low-perplexity) GCG for the defense loop
  5. eval      Unified per-(person,field) metric, paired stats, linguistic analysis
  6. defense   Input/perplexity/output filters vs naive AND adaptive adversary + benign FPR
  7. ablation  Prompt length, fluency-lambda sweep, convergence

Usage:
  python run_experiments.py                 # everything
  python run_experiments.py --stage attack  # a single stage
  python run_experiments.py --stage adaptive
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    train_cfg, eval_cfg, data_cfg, gcg_cfg, baseline_cfg,
    DATA_DIR, MODEL_DIR, RESULTS_DIR, DEVICE, DEVICE_PROFILE,
)


def _load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


def _free(model):
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def _get_model_paths():
    paths = {}
    for m in train_cfg.get_models():
        safe = m.replace("/", "_")
        path = os.path.join(MODEL_DIR, safe)
        if os.path.exists(os.path.join(path, "config.json")):
            paths[m] = path
    return paths


def _load_results(tag, safe, seed):
    path = os.path.join(RESULTS_DIR, f"{tag}_{safe}_seed{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_data():
    print("\n" + "#" * 70 + "\n# STAGE 1: DATA GENERATION\n" + "#" * 70)
    from data_generation import build_corpus
    return build_corpus()


def stage_train():
    print("\n" + "#" * 70 + "\n# STAGE 2: MODEL TRAINING\n" + "#" * 70)
    from train import train_all
    return train_all()


def stage_attack():
    print("\n" + "#" * 70 + "\n# STAGE 3: EXTRACTION ATTACKS\n" + "#" * 70)
    model_paths = _get_model_paths()
    if not model_paths:
        print("[ERROR] No trained models found. Run --stage train first.")
        return

    print("\n--- Baselines ---")
    from baselines import run_baselines_all_models, run_random_restart_all_models
    run_baselines_all_models(model_paths, eval_cfg.seeds)

    print("\n--- Naive GCG ---")
    from gcg_attack import run_gcg_all_models
    run_gcg_all_models(model_paths, eval_cfg.seeds, fluency_lambda=0.0, tag="gcg")

    if baseline_cfg.include_random_restart_control:
        print("\n--- Compute-matched random-restart control ---")
        run_random_restart_all_models(model_paths, eval_cfg.seeds)


def stage_adaptive():
    """Fluency-regularized GCG: the low-perplexity adversary the defenses face."""
    print("\n" + "#" * 70 + "\n# STAGE 4: ADAPTIVE (FLUENT) GCG\n" + "#" * 70)
    model_paths = _get_model_paths()
    if not model_paths:
        print("[ERROR] No trained models found. Run --stage train first.")
        return
    from gcg_attack import run_gcg_all_models
    run_gcg_all_models(
        model_paths, eval_cfg.seeds,
        fluency_lambda=gcg_cfg.adaptive_fluency_lambda, tag="gcg_adaptive",
    )


def stage_eval():
    print("\n" + "#" * 70 + "\n# STAGE 5: EVALUATION & ANALYSIS\n" + "#" * 70)
    from evaluate import (
        evaluate_baseline_results, evaluate_gcg_results,
        aggregate_across_seeds, significance_test, generate_tables,
    )
    from linguistic_analysis import run_linguistic_analysis

    model_paths = _get_model_paths()
    all_model_results = {}

    for model_name in model_paths:
        safe = model_name.replace("/", "_")
        print(f"\n--- Evaluating {model_name} ---")

        base_seeds, gcg_seeds, rand_seeds = [], [], []
        for seed in eval_cfg.seeds:
            b = _load_results("baseline", safe, seed)
            g = _load_results("gcg", safe, seed)
            r = _load_results("random", safe, seed)
            if b is not None:
                base_seeds.append(evaluate_baseline_results(b))
            if g is not None:
                gcg_seeds.append(evaluate_gcg_results(g))
            if r is not None:
                rand_seeds.append(evaluate_gcg_results(r))

        if not base_seeds or not gcg_seeds:
            print(f"  [SKIP] Missing results for {model_name}")
            continue

        base_agg = aggregate_across_seeds(base_seeds)
        gcg_agg = aggregate_across_seeds(gcg_seeds)
        main_test = significance_test(base_agg, gcg_agg)

        entry = {"baseline": base_agg, "gcg": gcg_agg, "test": main_test}

        if rand_seeds:
            rand_agg = aggregate_across_seeds(rand_seeds)
            # Does GCG beat brute-force random search at equal budget?
            ctrl_test = significance_test(rand_agg, gcg_agg)
            entry["random_control"] = rand_agg
            entry["gcg_vs_random_test"] = ctrl_test
            print(f"  Random-restart EMR: {rand_agg['emr_mean']:.1f}%  "
                  f"GCG vs random ratio: {ctrl_test.get('ratio', 0):.2f}x "
                  f"(p={ctrl_test.get('p_value', 1):.4f})")

        all_model_results[safe] = entry

        ci = (f" [{base_agg.get('emr_ci_low', float('nan')):.1f},"
              f"{base_agg.get('emr_ci_high', float('nan')):.1f}]")
        print(f"  Baseline EMR: {base_agg['emr_mean']:.1f} ± {base_agg['emr_std']:.1f}%")
        print(f"  GCG EMR:      {gcg_agg['emr_mean']:.1f} ± {gcg_agg['emr_std']:.1f}%{ci}")
        print(f"  Ratio:        {main_test.get('ratio', 0):.2f}x "
              f"(McNemar p={main_test.get('p_value', 1):.4f}, "
              f"n_paired={main_test.get('n_paired', 0)})")
        print(f"  Neg-ctrl EMR: base {base_agg.get('negative_control_emr', 0):.1f}%, "
              f"gcg {gcg_agg.get('negative_control_emr', 0):.1f}%")

    if all_model_results:
        print("\n--- Summary tables ---")
        generate_tables(all_model_results)

    # Linguistic analysis on the first model (uses held-out reference model for ppl)
    first_model = next(iter(model_paths), None)
    if first_model:
        safe = first_model.replace("/", "_")
        model, tokenizer = _load_model(model_paths[first_model])
        b = _load_results("baseline", safe, eval_cfg.seeds[0])
        g = _load_results("gcg", safe, eval_cfg.seeds[0])
        print(f"\n--- Linguistic analysis (targets from {first_model}) ---")
        run_linguistic_analysis(model, tokenizer, b, g)
        _free(model)

    # Persist (drop the bulky raw records before saving the summary)
    def _slim(agg):
        return {k: v for k, v in agg.items() if k != "_pooled_records"}
    slim = {
        safe: {k: (_slim(v) if isinstance(v, dict) and "_pooled_records" in v else v)
               for k, v in entry.items()}
        for safe, entry in all_model_results.items()
    }
    with open(os.path.join(RESULTS_DIR, "final_results.json"), "w") as f:
        json.dump(slim, f, indent=2, default=str)
    print(f"\nFinal results saved to {os.path.join(RESULTS_DIR, 'final_results.json')}")


def stage_defense():
    print("\n" + "#" * 70 + "\n# STAGE 6: DEFENSE EXPERIMENTS\n" + "#" * 70)
    from defense_eval import run_all_defense_experiments, build_benign_queries

    model_paths = _get_model_paths()
    first_model = next(iter(model_paths), None)
    if not first_model:
        print("[ERROR] No trained models found. Run --stage train first.")
        return

    safe = first_model.replace("/", "_")
    seed0 = eval_cfg.seeds[0]
    b = _load_results("baseline", safe, seed0)
    g = _load_results("gcg", safe, seed0)
    adaptive = _load_results("gcg_adaptive", safe, seed0)

    if not b or not g:
        print("[ERROR] Missing attack results. Run --stage attack first.")
        return
    if not adaptive:
        print("[WARN] No adaptive-GCG results found (run --stage adaptive) — "
              "defense will be evaluated against the naive adversary only.")

    model, tokenizer = _load_model(model_paths[first_model])
    benign = build_benign_queries()
    run_all_defense_experiments(
        model, tokenizer,
        baseline_results=b, gcg_results=g,
        adaptive_gcg_results=adaptive, benign_queries=benign,
    )
    _free(model)


def stage_ablation():
    print("\n" + "#" * 70 + "\n# STAGE 7: ABLATION STUDIES\n" + "#" * 70)
    from gcg_attack import GCGAttack, format_target

    model_paths = _get_model_paths()
    first_model = next(iter(model_paths), None)
    if not first_model:
        print("[ERROR] No trained models found.")
        return

    model, tokenizer = _load_model(model_paths[first_model])

    with open(os.path.join(DATA_DIR, "target_registry.json")) as f:
        all_targets = json.load(f)
    targets = [t for t in all_targets
               if t["frequency"] >= 5 and not t["is_negative_control"]][:10]

    # Ablation 1: prompt suffix length
    print("\n--- Ablation: prompt suffix length ---")
    k_results = {}
    for k in [10, 15, 20, 25, 30]:
        succ = 0
        for t in targets:
            atk = GCGAttack(model, tokenizer, k=k, N=200)
            if atk.attack(format_target(t["person"], "email"), t["person"])["success"]:
                succ += 1
        k_results[k] = 100.0 * succ / max(len(targets), 1)
        print(f"    k={k}: {k_results[k]:.1f}%")

    # Ablation 2: fluency-lambda sweep (extraction vs suffix perplexity frontier)
    print("\n--- Ablation: fluency-lambda sweep ---")
    lam_results = {}
    for lam in [0.0, 0.05, 0.1, 0.2, 0.5]:
        succ, ppls = 0, []
        for t in targets:
            atk = GCGAttack(model, tokenizer, N=200, fluency_lambda=lam)
            res = atk.attack(format_target(t["person"], "email"), t["person"])
            succ += int(res["success"])
            ppls.append(res["suffix_perplexity"])
        lam_results[lam] = {
            "success_rate": 100.0 * succ / max(len(targets), 1),
            "median_suffix_ppl": float(sorted(ppls)[len(ppls) // 2]) if ppls else None,
        }
        print(f"    lambda={lam}: {lam_results[lam]['success_rate']:.1f}%  "
              f"median_suffix_ppl={lam_results[lam]['median_suffix_ppl']}")

    # Ablation 3: convergence (from the main naive-GCG logs)
    print("\n--- Ablation: convergence ---")
    safe = first_model.replace("/", "_")
    gcg_data = _load_results("gcg", safe, eval_cfg.seeds[0])
    convergence = {}
    if gcg_data:
        for ckpt in gcg_cfg.checkpoint_iterations:
            n_succ = n_tot = 0
            for t in gcg_data:
                for fr in t.get("field_results", {}).values():
                    n_tot += 1
                    if fr.get("success") and fr.get("iteration", 10**9) <= ckpt:
                        n_succ += 1
            convergence[ckpt] = 100.0 * n_succ / max(n_tot, 1)
            print(f"    iter {ckpt}: {convergence[ckpt]:.1f}%")

    with open(os.path.join(RESULTS_DIR, "ablation_results.json"), "w") as f:
        json.dump({"prompt_length": k_results,
                   "fluency_lambda": lam_results,
                   "convergence": convergence}, f, indent=2, default=str)
    _free(model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_STAGES = ["data", "train", "attack", "adaptive", "eval", "defense", "ablation"]


def main():
    parser = argparse.ArgumentParser(description="PII Extraction Experiment Pipeline")
    parser.add_argument("--stage", choices=_STAGES + ["all"], default="all")
    args = parser.parse_args()

    print("=" * 70)
    print("PII EXTRACTION EXPERIMENT PIPELINE")
    print(f"Device: {DEVICE} | Profile: {DEVICE_PROFILE}")
    print(f"Models: {train_cfg.get_models()} | Seeds: {eval_cfg.seeds}")
    print("=" * 70)

    t0 = time.time()
    run = _STAGES if args.stage == "all" else [args.stage]
    dispatch = {
        "data": stage_data, "train": stage_train, "attack": stage_attack,
        "adaptive": stage_adaptive, "eval": stage_eval,
        "defense": stage_defense, "ablation": stage_ablation,
    }
    for s in run:
        dispatch[s]()

    print(f"\n{'='*70}\nPipeline complete. Total time: {(time.time()-t0)/3600:.1f} h")
    print(f"Results in: {RESULTS_DIR}\n{'='*70}")


if __name__ == "__main__":
    main()
