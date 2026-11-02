# Experiment 4: Steer2Edit

Steer2Edit (Sun, Yan, Wang, Weng, arXiv:2602.09870) turns a steering vector into **rank-1
weight edits** on individual attention heads (`o_proj` column blocks) and MLP neurons
(`down_proj` columns). It adds no hooks at inference: once the edit is made, the model is an
ordinary model with slightly different weights. This port follows the released code
(`github.com/Trustworthy-ML-Lab/Steer2Edit`, `safety_alignment/extract_directions.py` and
`edit.py`). It reuses this repo's model loading, chat template, eval prompts, greedy decoding
and `is_refusal`, so its numbers line up with the `baseline` / `fixed-addition` / `CLAS` table.

## Method

For each edited layer *l* and each block type *b* ∈ {attn, mlp}:

```
v_l^b = mean(block output | refusal responses) − mean(block output | compliant responses)
        (averaged over response tokens, then over samples)

per component i (head or neuron), with weight slice W_i and mean input μ_i:
  g_i     = cos(v, W_i μ_i)                                   importance
  λ_i     = sign(g_i) · max(|g_i| − ρα, 0) / (ρ(1 − α))       elastic-net closed form
  k̂_i     = W_iᵀ v / ‖W_iᵀ v‖
  ΔW_i    = λ_i · v̂ k̂_iᵀ
```

`ρ_attn` and `ρ_mlp` are budgets (a larger ρ gives a smaller, sparser edit). α sets the
trade-off between L1 and L2, and `ρ = −1` turns that block type off.

## Running

```bash
cd refusal-steering
python run_steer2edit.py --smoke                 # wiring check, a couple of minutes
python run_steer2edit.py                         # main result: layers 16-23, like Exps 1/3
```

Optional variants, each written to its own `outputs_steer2edit_<tag>/`:

```bash
python run_steer2edit.py --layers all --tag all_layers   # paper-faithful: edit every layer
python run_steer2edit.py --direction arditi --tag arditi # v = main.py's residual r̂ (needs outputs/r_hat.pt)
python run_steer2edit.py --random-direction --tag random # control: random unit vectors
python run_steer2edit.py --grid-rho-attn 0.4,0.5,0.6 --grid-alpha 0.6,0.7,0.8 --tag refine  # stage-2 grid
```

The grid is checkpointed to `grid_sweep.csv` after every config. If the job is killed, rerun
the same command and it resumes where it stopped. The default grid is the paper's stage-1
coarse grid: ρ_attn and α in {0.1, 0.3, 0.5, 0.7, 0.9}, ρ_mlp in {off, 0.1, …, 0.9}. That is
150 configs at 50 + 50 validation prompts each, about 1–1.5 h on one A100, plus about 5 min
for directions and the final evaluation.

Unit tests (CPU, no download, tiny random Llama):
`PYTHONPATH=.:tests python tests/test_steer2edit_math.py`. They check the batched span
masking against an unpadded reference, every λ / k̂ against the formula computed
independently, that the edits are exactly rank-1 along v̂, and that the weights are restored
bit-exactly.

## Outputs (`outputs_steer2edit/`)

| file | content |
|---|---|
| `results_table.csv`, `derived_metrics.csv` | same columns as `main.py`; conditions `baseline`, `SteerToEdit` |
| `generations_{baseline,SteerToEdit}_{risky,benign}.csv` | every generation, for spot-checks |
| `grid_sweep.csv` | val ASR / over-refusal / degeneracy / #heads / #neurons for each config |
| `s2e_directions.pt`, `s2e_edits.pt` | steering vectors and the selected edit (official pack format) |
| `responses_{positive,negative}.csv` | the model's own responses the vectors were built from |
| `run_config.json` | selected (ρ_attn, ρ_mlp, α), the selection rule, edit statistics per layer |

## Design choices and deviations

1. **Layers 16–23 by default.** The paper edits whatever every layer's λ allows. Here the
   default matches Exps 1 and 3, so all three arms touch the same layers. `--layers all` is
   the paper's setting.
2. **Directions come from the method's own vectors, not r̂.** Steer2Edit's vector is a
   per-block output difference over response tokens. Arditi's r̂ is a residual-stream
   difference at the last prompt token. Both are built from AdvBench vs Alpaca, and no
   medical data is used. `--direction arditi` runs the same edit rule with r̂, which
   isolates "how the direction is applied" from "which direction". The log prints cos(v, r̂)
   for each layer.
3. **The released code's pairing.** `extract_directions.py` puts the compliant (Alpaca)
   responses under the *harmful* questions, so the contrast is refusal vs. compliance under
   the same kind of prompt. That is the default (`--pairing official`). `--pairing own` puts
   each response under its own question instead.
4. **Positives and negatives are filtered.** Only AdvBench responses that `is_refusal`
   counts as refusals become positives, and only Alpaca responses it doesn't count become
   negatives. The released code doesn't filter.
5. **Greedy decoding, 128-token responses, no system prompt.** The released code samples up
   to 1024 tokens and uses the Llama-2 safety system prompt. Here everything matches this
   repo's eval setup.
6. **Selection on held-out validation.** The grid is scored on MedSafetyBench *train* and
   MedQuAD *train-group* prompts, which are disjoint from the 200 + 200 eval prompts. Among
   configs with val ASR ≥ 0.90 and degeneracy ≤ 0.10, the one with the lowest val
   over-refusal is chosen. A 3-D grid has no "smallest α", so this is the closest analogue
   of `main.py`'s rule. Note that `main.py`'s α sweep scores on a prefix of the eval
   prompts, so its selection is not held out.
7. **Llama-3.1-8B is not in the paper.** It reports refined ranges only for Llama-2-7B
   (ρ_attn 0.16–0.24, ρ_mlp 0.35–0.55, α 0.70–0.90) and Mistral-7B. Hence the coarse grid
   first, then an optional refined grid around the winner.
8. **Same caveat on the metric as the other arms.** Small ρα gives large λ on most
   components and can collapse the model. Degeneracy is reported for every config, and a
   collapsed config is never selected.
