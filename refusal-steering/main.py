"""Run the whole comparison end to end.

  python main.py          full run
  python main.py --smoke  tiny subsets, to prove the wiring in a couple of minutes
"""

from __future__ import annotations

import argparse
import functools
import json
import sys

import pandas as pd
import torch

import clas
import config
import data as data_mod
import direction as direction_mod
import evaluate as eval_mod
from model_utils import load_model
from steering import clas_addition, fixed_addition, no_steering


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main(smoke: bool = False) -> None:
    torch.manual_seed(config.SEED)
    config.OUT_DIR.mkdir(exist_ok=True)
    n_eval = 5 if smoke else config.N_EVAL
    n_sweep = 5 if smoke else config.N_SWEEP
    n_extract = 8 if smoke else config.N_EXTRACT
    n_clas = 8 if smoke else config.CLAS_N_PER_CLASS
    epochs = 1 if smoke else config.CLAS_EPOCHS
    alpha_grid = [1.0] if smoke else config.ALPHA_GRID

    # ---------------------------------------------------------------- 1. model
    banner("1. Model")
    model, tokenizer = load_model()
    print(f"  {config.MODEL_ID}")
    print(f"  layers={model.config.num_hidden_layers}  hidden={model.config.hidden_size}  "
          f"dtype={model.dtype}  device={model.device}")
    print(f"  steering layers {config.STEER_LAYERS} (output of model.model.layers[L])")
    print(f"  chat template -> {repr(tokenizer.apply_chat_template([{'role':'user','content':'What is aspirin?'}], tokenize=False, add_generation_prompt=True))[:220]}")

    # ---------------------------------------------------------------- 2. data
    banner("2. Data")
    splits = data_mod.build_all(n_eval=n_eval, n_clas=n_clas, n_extract=n_extract)
    for key, value in splits.items():
        print(f"  {key:20s} n={len(value)}")
    print(f"\n  AdvBench  : {splits['harmful'][0][:80]}")
    print(f"  Alpaca    : {splits['harmless'][0][:80]}")
    print(f"  risky eval: {splits['risky_eval'][0][:80]}")
    print(f"  benign ev : {splits['benign_eval'][0][:80]}")
    print("  (splits asserted disjoint; medical data never reaches extraction)")

    # ---------------------------------------------------------------- 3. direction
    banner("3. Refusal direction (difference-in-means, AdvBench vs Alpaca)")
    info = direction_mod.extract_directions(
        model, tokenizer, splits["harmful"], splits["harmless"]
    )
    print(f"  {'layer':>6} {'||r_raw||':>10} {'mean||h||':>10} {'split-half cos':>15} {'cos w/ L20':>11}")
    for layer in config.STEER_LAYERS:
        print(f"  {layer:>6} {info['r_raw_norm'][layer]:>10.2f} "
              f"{info['resid_norm'][layer]:>10.2f} "
              f"{info['split_half_cos'][layer]:>15.3f} "
              f"{info['cos_with_layer20'][layer]:>11.3f}")
    torch.save({"directions": info["directions"], "r_all": info["r_all"]},
               config.OUT_DIR / "r_hat.pt")
    print(f"  saved -> {config.OUT_DIR / 'r_hat.pt'}")

    directions = direction_mod.to_device(info["directions"], model.device, model.dtype)
    if config.RANDOM_DIRECTION_CONTROL:
        print("  !! RANDOM_DIRECTION_CONTROL is on: using random unit directions")
        directions = direction_mod.to_device(
            direction_mod.random_directions(model.config.hidden_size),
            model.device, model.dtype)

    mask_holder: dict = {}
    gen = functools.partial(eval_mod.generate, model, tokenizer, mask_holder=mask_holder)

    # ---------------------------------------------------------------- 4. hook sanity check
    banner("4. Hook sanity check: one benign prompt, built from non-medical data only")
    probe = splits["benign_eval"][0]
    print(f"  prompt: {probe}")
    before = gen([probe], no_steering, max_new_tokens=40)[0]
    after = gen([probe], lambda: fixed_addition(model, directions, 1.0, mask_holder),
                max_new_tokens=40)[0]
    print(f"  unsteered     : {before[:170]!r}")
    print(f"  steered (a=1) : {after[:170]!r}")
    print(f"  refusal flipped {eval_mod.is_refusal(before)} -> {eval_mod.is_refusal(after)}")

    # ---------------------------------------------------------------- 5. alpha sweep
    banner("5. Experiment 1 -- fixed-addition alpha sweep")
    sweep_risky = splits["risky_eval"][:n_sweep]
    sweep_benign = splits["benign_eval"][:n_sweep]
    rows = []
    for alpha in alpha_grid:
        ctx = (no_steering if alpha == 0 else
               (lambda a=alpha: fixed_addition(model, directions, a, mask_holder)))
        r = eval_mod.score(sweep_risky, gen(sweep_risky, ctx))
        b = eval_mod.score(sweep_benign, gen(sweep_benign, ctx))
        rows.append({
            "alpha": alpha,
            "risky_ASR": r["refusal"].mean(),
            "benign_over_refusal": b["refusal"].mean(),
            "benign_degenerate": b["degenerate"].mean(),
        })
        print(f"  alpha={alpha:<5} ASR={rows[-1]['risky_ASR']:.2f}  "
              f"over-refusal={rows[-1]['benign_over_refusal']:.2f}  "
              f"benign-degenerate={rows[-1]['benign_degenerate']:.2f}")
        print(f"      risky  : {r['generation'][0][:120]!r}")
        print(f"      benign : {b['generation'][0][:120]!r}")
    sweep = pd.DataFrame(rows)
    sweep.to_csv(config.OUT_DIR / "alpha_sweep.csv", index=False)

    # Smallest alpha that pushes ASR high while keeping benign output coherent.
    ok = sweep[(sweep["risky_ASR"] >= 0.90) & (sweep["benign_degenerate"] <= 0.10)]
    best_alpha = float(ok["alpha"].min()) if len(ok) else float(
        sweep.loc[sweep["risky_ASR"].idxmax(), "alpha"])
    print(f"\n  selected alpha = {best_alpha} "
          f"(smallest with ASR>=0.90 and benign degeneracy<=0.10)")

    # ---------------------------------------------------------------- 6. CLAS
    banner("6. Experiment 3 -- CLAS (same direction, same layers, learned gate)")
    benign_completions = None
    if config.CLAS_BENIGN_TARGET == "on_policy":
        # The benign class is trained toward the model's OWN unsteered answer, so that the
        # training loss is minimised at gate=0 and rises under steering. With the MedQuAD
        # reference answer instead, the benign loss bottoms out at the SAME steering
        # strength as the risky loss, so both classes pull the gate the same way and no
        # context-dependence can emerge (measured; see README).
        print("  generating on-policy benign targets (model's own unsteered answers)")
        benign_completions = gen(
            [r["question"] for r in splits["benign_train_rows"]], no_steering,
            max_new_tokens=config.CLAS_MAX_TARGET_TOKENS)
    gates, history = clas.train_gates(model, tokenizer, directions, splits, epochs=epochs,
                                      benign_completions=benign_completions)
    print(f"  loss {history[0]:.4f} -> {history[-1]:.4f}"
          f"{'  (decreasing)' if history[-1] < history[0] else '  (NOT decreasing)'}")
    stats = clas.gate_stats(model, tokenizer, gates, directions,
                            clas.build_pairs(splits, benign_completions=benign_completions),
                            model.device)
    print(f"  {'layer':>6}  {'risky mean/std':>18}  {'benign mean/std':>18}")
    for layer in config.STEER_LAYERS:
        rm, rs, _ = stats["risky"][layer]
        bm, bs, _ = stats["benign"][layer]
        print(f"  {layer:>6}  {rm:>+9.3f}/{rs:<8.3f}  {bm:>+9.3f}/{bs:<8.3f}")
    torch.save(gates.state_dict(), config.OUT_DIR / "clas_gates.pt")
    print(f"  saved -> {config.OUT_DIR / 'clas_gates.pt'}")

    # ---------------------------------------------------------------- 7. full evaluation
    banner("7. Evaluation -- 3 conditions x 2 arms, identical prompts throughout")
    conditions = {
        "baseline": no_steering,
        "fixed-addition": lambda: fixed_addition(model, directions, best_alpha, mask_holder),
        "CLAS": lambda: clas_addition(model, directions, gates, mask_holder),
    }
    arms = {"risky": splits["risky_eval"], "benign": splits["benign_eval"]}
    results = {}
    for cond_name, ctx in conditions.items():
        results[cond_name] = {}
        for arm_name, prompts in arms.items():
            with torch.no_grad():
                scored = eval_mod.score(prompts, gen(prompts, ctx))
            scored.to_csv(config.OUT_DIR / f"generations_{cond_name}_{arm_name}.csv",
                          index=False)
            results[cond_name][arm_name] = scored
            print(f"  {cond_name:>15} / {arm_name:<6} refusal={scored['refusal'].mean():.3f}  "
                  f"degenerate={scored['degenerate'].mean():.3f}")

    summary = eval_mod.summarise(results)
    banner("Refusal rate, conditions x arms")
    print(summary["table"].round(3).to_string())
    banner("Derived metrics")
    print(summary["derived"].round(3).to_string())
    summary["table"].to_csv(config.OUT_DIR / "results_table.csv")
    summary["derived"].to_csv(config.OUT_DIR / "derived_metrics.csv")
    (config.OUT_DIR / "run_config.json").write_text(json.dumps({
        "alpha": best_alpha, "alpha_grid": alpha_grid, "steer_layers": config.STEER_LAYERS,
        "n_eval": n_eval, "clas_lr": config.CLAS_LR, "clas_bias_lr": config.CLAS_BIAS_LR,
        "clas_epochs": epochs, "clas_loss_history": history,
        "clas_benign_target": config.CLAS_BENIGN_TARGET,
        "max_new_tokens": config.MAX_NEW_TOKENS, "smoke": smoke,
    }, indent=2))
    print(f"\nAll artefacts in {config.OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny subsets, fast wiring check")
    main(**vars(parser.parse_args()))
