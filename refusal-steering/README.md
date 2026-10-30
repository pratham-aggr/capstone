# Refusal steering on Llama-3.1-8B-Instruct: fixed addition vs. CLAS

One refusal direction, extracted from general non-medical data, applied two ways to the
same layers of the same model:

* **Experiment 1 — fixed activation addition.** `h ← h + α·r̂_ℓ`. One scalar α steers every
  prompt identically, so it cannot tell a risky medical request from a benign one.
* **Experiment 3 — CLAS (context-gated addition).** `h ← h + (c_ℓ·[h,1])·r̂_ℓ`, where the
  steering strength is a learned affine function of the activation itself, so it *can*
  depend on context.

Both use the **same directions** and the **same layers (16–23)**. The only difference is
fixed scalar vs. learned gate — that contrast is the entire point of the repository.

The question is not just "can we induce refusals" but **what does it cost on benign
prompts**: refusing a genuinely risky medical request is the goal, refusing "What are the
treatments for X?" is collateral damage.

## Layer / activation convention

> **"Residual stream at layer ℓ" = the OUTPUT of `model.model.layers[ℓ]`**, equivalently
> `output_hidden_states[ℓ+1]`.

Extraction and every intervention use this one convention, so a direction measured at
layer ℓ is added back to exactly the tensor it was measured from.

## Data — each source used for one purpose only

| Source | Purpose | Notes |
|---|---|---|
| AdvBench `harmful_behaviors.csv` | direction: harmful | 520 rows, column `goal` |
| Alpaca `tatsu-lab/alpaca` | direction: harmless | rows with empty `input` only, so items are self-contained |
| MedSafetyBench `datasets/test` | **risky eval** | 900 prompts, column `harmful_medical_request` |
| MedSafetyBench `datasets/train` | CLAS risky-class training | disjoint from test upstream |
| MedQuAD `lavita/MedQuAD` | **benign eval** + CLAS benign-class training | disjoint question groups |

The medical datasets are **never** used for direction extraction, and AdvBench/Alpaca are
**never** evaluated on. The direction is therefore independent of the evaluation
distribution, and a benign medical prompt flipping to a refusal is a genuine transfer
result.

**MedQuAD needs care.** 31,034 of its 47,441 rows have empty answers (removed for
copyright), leaving 16,407 usable. Those 16,407 questions cover only ~3,500 conditions, so
a row-level split would put "What is X?" in training and "How many people are affected by
X?" in eval. We group by normalised `question_focus` (falling back to `document_id`), split
the **groups**, and draw **one question per group** for the eval arm.

## Metric — ASR by substring matching

`is_refusal` normalises both the generation and each marker by replacing the typographic
apostrophe `’` (U+2019) with ASCII `'` and lowercasing, then tests substring containment.
The marker list is verbatim from Arditi et al. (2024); it is typed with `’` while the model
emits `'`, so **without the normalisation roughly half the markers would never match**.

The refusal rate is measured *identically* on both arms and only then labelled by role:

* **risky arm refusal rate = ASR** — higher is better, this is the goal
* **benign arm refusal rate = over-refusal** — lower is better, this is the damage

Because the model already refuses some risky prompts unprompted, we also report:

* `induced ASR = ASR(steered) − ASR(baseline)`
* `induced ASR on the no-headroom subset` — ASR among only the risky prompts the model did
  **not** refuse at baseline. The subset is fixed once from the baseline run and reused for
  every condition. It is small, so it is reported with its denominator.

### Known limitations of the metric

* It catches **explicit** refusals only, not soft non-answers or topic changes.
* A medical disclaimer that still answers ("…but consult a doctor") correctly does **not**
  count as a refusal.
* It is blind to **degeneration**. Strong steering can collapse the model into repetitive
  gibberish, which contains no marker and so scores as "not a refusal" — making
  over-refusal read *low* exactly when the model has been destroyed. Every cell therefore
  also reports `looks_degenerate`, and no steering strength is selected on a cell whose
  benign output is incoherent. This is a guardrail, not a headline metric.

All generations are written to `outputs/generations_{condition}_{arm}.csv` for
hand spot-checking.

## Results

200 prompts per arm, identical prompts in every cell, greedy decoding, 64 new tokens.
Steering layers 16-23, one shared set of per-layer directions, alpha selected by the sweep
as the smallest value reaching ASR >= 0.90 with benign degeneracy <= 0.10.

| condition | risky ASR | benign over-refusal | induced ASR | no-headroom ASR (n=51) | selectivity |
|---|---|---|---|---|---|
| baseline | 0.745 | 0.035 | — | — | 0.710 |
| fixed-addition (α=1.0) | 0.970 | **0.720** | +0.225 | 0.882 | 0.250 |
| CLAS | 0.950 | **0.110** | +0.205 | 0.804 | **0.840** |

Degeneracy was 0.000 in every cell, so no number here is an artefact of a collapsed model.

The contrast is the point: **fixed addition and CLAS reach almost the same ASR (0.970 vs
0.950), but fixed addition refuses 72% of benign medical questions while CLAS refuses
11%.** A single scalar cannot tell "how do I justify an unnecessary surgery" from "what are
the treatments for X", so buying refusals on the first costs refusals on the second. A gate
that reads the activation can separate them.

The alpha sweep shows the trade-off is not a tuning artefact — every fixed coefficient
sits on the same frontier:

| alpha | risky ASR | benign over-refusal | benign degenerate |
|---|---|---|---|
| 0.25 | 0.80 | 0.16 | 0.00 |
| 0.50 | 0.84 | 0.30 | 0.00 |
| 0.75 | 0.88 | 0.58 | 0.00 |
| 1.00 | 0.94 | 0.78 | 0.00 |
| 1.50 | 0.98 | 0.94 | 0.08 |
| 2.00 | 1.00 | 0.98 | 0.72 |

There is no coefficient that buys high ASR cheaply. CLAS lands off this frontier.

Note the last row: at alpha=2.0 the model is 72% degenerate, emitting repetitive fragments.
Its over-refusal reads 0.98 here, but push alpha to 4 and both arms read **0.00** refusal
purely because the output is gibberish — which is why `looks_degenerate` is reported beside
every rate.

## Layout

```
config.py       every hyperparameter, REFUSAL_TARGET, REFUSAL_MARKERS
data.py         the four loaders, filters and disjoint splits
direction.py    difference-in-means extraction + diagnostics
steering.py     the forward hooks (fixed addition, CLAS gate)
clas.py         gate training
evaluate.py     generation, is_refusal, ASR / over-refusal / induced ASR
model_utils.py  model loading and chat formatting shared by the above
main.py         end to end
```

Plain `torch` forward hooks throughout — no interpretability framework, so the mechanism
stays visible.

## Running

MedSafetyBench is cloned by hand (it is not vendored into this repo):

```bash
cd refusal-steering
mkdir -p data_raw && git clone https://github.com/AI4LIFE-GROUP/med-safety-bench.git data_raw/med-safety-bench
```

AdvBench is fetched over HTTP at runtime; Alpaca and MedQuAD come from the HuggingFace
`datasets` cache. Then:

```bash
export CUDA_VISIBLE_DEVICES=3
python main.py --smoke   # tiny subsets, proves the wiring in a couple of minutes
python main.py           # full run
```

`meta-llama/Llama-3.1-8B-Instruct` is gated; a valid `HF_TOKEN` must be set (or the model
already cached). The code stops with an explicit error rather than falling back to another
model.

## Deliberate deviations from the source papers

1. **Per-layer directions.** The spec called for one `r̂` reused everywhere. Measurement
   showed the difference-in-means direction **rotates substantially with depth**:
   `cos(r̂@11, r̂@16…23)` is only 0.08–0.24, while `cos(r̂@20, r̂@16…23)` is 0.51–0.82. Arditi
   et al.'s chosen layer 12 is valid *because they add at layer 12* — extraction and
   intervention coincide there. Transplanting that index into a setup that steers at 16–23
   would apply a near-orthogonal vector, so instead we extract a separate unit direction at
   each steered layer from the same single forward pass. **Both experiments use the
   identical set of directions**, so the fixed-vs-gated contrast is untouched. This is also
   closer to CLAS, which uses layer-specific `d_ℓ`.
2. **Layers 16–23 rather than all blocks.** CLAS steers every block; here both arms must
   use the same layers for the comparison to mean anything.
3. **Unit directions with an explicit α.** Arditi et al. add the *unnormalised* `r` at a
   *single* layer, its norm serving as the coefficient. Adding across 8 layers compounds,
   so α is swept.
4. **Learning rate.** The CLAS paper's `lr=3e-3` (with a separate larger `1e-1` on the bias
   entry) is calibrated to their setup; here it drove the gate past the point where the
   model degenerates, and the tripwire in `steering.py` caught it. The value actually used
   is recorded in `outputs/run_config.json`.
5. **The benign training target is the model's own answer, not the MedQuAD reference.**
   See below — this one changes whether the experiment works at all.

### Why the benign CLAS target had to change

The original design trained the benign class toward its **MedQuAD reference answer**. That
does not work, and the reason is measurable rather than a matter of tuning. Sweeping a
*constant* gate value and recording the training loss of each class:

| gate | risky loss (canonical refusal) | benign loss (MedQuAD ref) | benign loss (on-policy) |
|---|---|---|---|
| 0.0 | 2.149 | 2.231 | **1.024** |
| 0.5 | 2.119 | 2.148 | 0.999 |
| 1.0 | **1.970** | **2.137** | 1.053 |
| 1.5 | 2.060 | 2.269 | 1.189 |
| 2.0 | 2.422 | 2.629 | 1.497 |

With the reference answer, the benign loss is **minimised at the same steering strength
that minimises the risky loss** — both bottom out near gate ≈ 1. The two classes therefore
pull the gate in the *same* direction, nothing rewards leaving benign prompts alone, and no
amount of training can produce context-dependence. Empirically the trained gate then
over-refused benign prompts and *reduced* risky ASR below baseline.

The cause is that a MedQuAD answer is text the model would never naturally produce, so its
loss at gate = 0 (2.23) sits far above its floor; the gate cannot close that gap with one
scalar per layer, and the residual gradient is noise. Shortening the target to 16 or 12
tokens does not help — the mismatch is in style, not length.

Training the benign class toward the **model's own unsteered answer** fixes the signal: the
loss is near its floor at gate = 0 and rises steeply with steering, so the benign class now
actively pushes the gate toward zero while the risky class pushes it up. That is the
contrast the gate needs in order to learn context at all.

`config.CLAS_BENIGN_TARGET` selects between the two (`"reference"` reproduces the original
behaviour), so the failure is reproducible rather than merely asserted.

## Reading the CLAS result honestly

`REFUSAL_TARGET` deliberately begins with `"I'm sorry"`, which is itself one of the
detector's markers — the spec requires the training target and the metric to agree.
A consequence is that **CLAS's risky-arm ASR is not independent evidence**: the gate is
trained to emit the exact string the metric looks for. The genuinely informative numbers
are therefore:

* **benign over-refusal** — never trained toward a refusal, so this is uncontaminated; and
* **risky ASR on the held-out MedSafetyBench *test* split**, which the gate never saw.

A gate that simply always fires would score high ASR *and* high over-refusal and be
worthless, which is why `selectivity = ASR − over-refusal` is reported alongside.

With `CLAS_RISKY_TARGET = "steered_self"` the risky class is trained toward the fixed
steer's own refusals, so CLAS is in effect asked to **reproduce fixed addition's behaviour
on risky prompts while leaving benign ones alone**. Its risky ASR is therefore partly a
distillation result and should not be read as independent evidence that steering induces
refusal — the fixed-addition arm already establishes that. The clean, uncontaminated
comparison is the **benign arm**, where the two methods differ by 0.72 vs 0.11 and where
CLAS was never trained toward a refusal at all. Training uses the MedSafetyBench *train*
split; every number above is on the disjoint *test* split.

## Citations

* Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, NeurIPS 2024 (arXiv:2406.11717)
* Hsu, Beaglehole, Radhakrishnan, Belkin, *Contextual Linear Activation Steering of Language Models* (arXiv:2604.24693)
* Han, Kumar, Agarwal, Lakkaraju, *MedSafetyBench*, NeurIPS 2024 D&B (arXiv:2403.03744)
