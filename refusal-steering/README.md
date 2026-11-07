# Fixed activation addition vs. CLAS on Llama-3.1-8B-Instruct

One refusal direction, one set of layers. The only difference between the two
experiments is how the steering coefficient is chosen:

- **Fixed addition:** `h += alpha * r_hat`, same `alpha` for every prompt.
- **CLAS:** `h += (c · [h, 1]) * r_hat`, coefficient learned from the prompt.

Question: does context-gating avoid the collateral damage that a fixed coefficient
causes on benign prompts?

## Setup

| | |
|---|---|
| model | `meta-llama/Llama-3.1-8B-Instruct`, bf16, single GPU |
| direction | difference-in-means, AdvBench vs. Alpaca, last token, unit norm |
| steered layers | 16–23, all token positions |
| risky eval | MedSafetyBench test split, 200 prompts |
| benign eval | MedQuAD, 200 prompts |
| metric | refusal-marker substring match; 64 new tokens, greedy |

Medical data is never used to extract the direction. AdvBench/Alpaca are never used
for eval. Train and eval splits are disjoint.

## Results (200 prompts per arm)

Risky ASR: higher is better. Benign over-refusal: lower is better.

| condition | risky ASR | benign over-refusal | selectivity |
|---|---|---|---|
| baseline | 0.745 | 0.035 | 0.71 |
| fixed addition (alpha=1.0) | 0.970 | **0.720** | 0.25 |
| CLAS | 0.950 | **0.110** | **0.84** |

Degeneracy was 0.000 in every cell, so no number above is an artifact of broken output.

**Both methods push risky refusal to ~0.96. Only the fixed coefficient drags benign
refusal from 0.035 to 0.72.** CLAS gets the same safety gain for 1/7th the collateral
cost.

### Alpha sweep (50 prompts per arm)

There is no alpha that buys risky refusal without paying on benign prompts:

| alpha | risky ASR | benign over-refusal | benign degenerate |
|---|---|---|---|
| 0.25 | 0.80 | 0.16 | 0.00 |
| 0.50 | 0.84 | 0.30 | 0.00 |
| 0.75 | 0.88 | 0.58 | 0.00 |
| **1.00** | 0.94 | 0.78 | 0.00 |
| 1.50 | 0.98 | 0.94 | 0.08 |
| 2.00 | 1.00 | 0.98 | 0.72 |

Above alpha=2 the model emits gibberish and *both* refusal rates read 0.00 — which
looks like success if coherence is not checked. Hence the degeneracy column.

### Control: random direction (`outputs_random/`)

Same layers, same alphas, a random vector of matched norm instead of `r_hat`:

| condition | risky ASR | benign over-refusal |
|---|---|---|
| baseline | 0.745 | 0.035 |
| fixed addition | 0.740 | 0.040 |
| CLAS | 0.735 | 0.050 |

Flat at baseline everywhere. The effect comes from the refusal direction, not from
perturbing the residual stream, and CLAS cannot manufacture refusal through an
arbitrary vector.

### Variant: canonical refusal target for CLAS (`outputs_canonical/`)

Training the gate to emit the fixed string `"I'm sorry, but I can't help with that
request."` on risky prompts **fails**: risky ASR 0.505 (below baseline 0.745),
over-refusal 0.075. The gate learns the refusal *style* while turning 53 of 149
baseline refusals into compliances; the mean gate at layer 20 goes negative, i.e. it
subtracts the refusal direction. Using the model's own steered output as the target
(the main result above) fixes this.

## Caveats

- CLAS's risky ASR is partly circular: it is trained toward refusal-shaped text and
  scored by a refusal-marker matcher. The informative number is benign over-refusal.
- The matcher catches explicit refusals only, not soft non-answers. All generations
  are saved for spot-checking.
- Only 200 prompts per arm; the no-headroom subset is n=51.

## AI and human contributions

The research question, experiment plan and design decisions were made by Amrutha Potluri and
Pratham Aggarwal. The code was written with AI assistance (Claude, Anthropic) from our
prompts; we reviewed it, tested it, and ran and analysed the experiments. Key prompts and
how the code was tested are documented in [`AI_USE.md`](AI_USE.md).

## Running

```bash
pip install torch transformers datasets numpy pandas
git clone https://github.com/AI4LIFE-GROUP/med-safety-bench data_raw/med-safety-bench
export HF_TOKEN=...                  # Llama-3.1 is gated

python main.py --smoke               # fast wiring check
python main.py                       # main result            -> outputs/
python main.py --tag random --random-direction                -> outputs_random/
python main.py --tag canonical --clas-risky-target canonical  -> outputs_canonical/
```

Every hyperparameter is in `config.py`.

## Layout

```
config.py      hyperparameters
data.py        loaders + disjoint splits
direction.py   difference-in-means direction
steering.py    forward hooks: fixed addition, CLAS gate
clas.py        gate training
evaluate.py    generation, refusal matcher, metrics
main.py        end-to-end, 7 stages
```

Each output dir holds `results_table.csv`, `derived_metrics.csv`, `alpha_sweep.csv`,
`run_config.json`, and the full generations per condition and arm.

## References

- Arditi et al., *Refusal in LLMs is mediated by a single direction*, arXiv:2406.11717
- Hsu, Beaglehole, Radhakrishnan, Belkin, *CLAS*, arXiv:2604.24693
- Han et al., *MedSafetyBench*, NeurIPS 2024 D&B, arXiv:2403.03744
