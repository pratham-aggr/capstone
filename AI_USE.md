# AI use

AI tools were used in two parts of this project: writing and testing the code, and writing the report (ChatGPT). This file records both.

## Summary of contributions

### Code and experiments

| Work | Human | AI |
| --- | --- | --- |
| Research question, experiment plan (baseline → fixed addition → CLAS → Steer2Edit) | ✔ |  |
| Choice of Steer2Edit as experiment 4 | ✔ |  |
| Finding and reading the Steer2Edit paper and official code |  | ✔ |
| `steer2edit.py`, `run_steer2edit.py`, `STEER2EDIT.md`, `run_baseline.py` | reviewed | written |
| `tests/` (CPU unit tests) | ran them on own laptop | written |
| Baseline analysis (CIs, per-category rates, knowledge-gap finding) | reviewed | computed from existing outputs |
| Design decisions (layers 16–23, held-out selection, following the authors' code) | approved | proposed |
| Experiments 1–3 design: data roles and disjoint splits, layers 16–23, per-layer directions, refusal metric with degeneracy check | ✔ |  |
| Experiments 1–3 code (`main.py`, `direction.py`, `steering.py`, `clas.py`, `data.py`, `evaluate.py`) | directed (prompts 4–6) and reviewed | assisted |
| Diagnosing and fixing CLAS training targets (on-policy benign target, steered-self risky target, lower learning rate) | ✔ |  |
| Running experiments 1–3 on the GPU and analysing the results | ✔ |  |

### Report (ChatGPT)

| Work | Human | AI |
| --- | --- | --- |
| Introduction: first draft | ✔ |  |
| Introduction: grammar and wording polish (prompt R1) | requested | ✔ |
| Literature review: content and sources | ✔ |  |
| Literature review: feedback on strengths and weaknesses (prompt R2) | requested | ✔ |
| Explanation and LaTeX equations for activation steering and CLAS, based on the cited papers (prompt R3) | requested | ✔ |
| Feedback on writing standards and cutting unneeded content (prompt R4) | requested | ✔ |
| Formatting the results table in LaTeX (prompt R5) | requested | ✔ |

## Key prompts and what the AI did

### Code

These are the important prompts that shaped the code, quoted as typed.

**1. "Using the existing repo, establish a baseline for Llama-3.1-8B-Instruct with no steering applied. Use the same eval prompts as `main.py` (200 MedSafetyBench risky, 200 MedQuAD benign), with the same greedy decoding and refusal metric. Report the refusal rate per arm with 95% confidence intervals, a breakdown by MedSafetyBench category, and examples of failure cases. Save the per-prompt results."**

The AI's environment had no GPU, so it could not load Llama-3.1-8B. It found that `main.py` had already produced unsteered baseline generations and re-analysed them: 95% CIs, the refusal phrases the model uses, refusal rate per MedSafetyBench category, and the non-refused prompts. It found that all 7 benign "refusals" are knowledge gaps ("I am unable to verify…"), not safety refusals. It wrote `run_baseline.py` so the baseline can be rerun on its own on a GPU.

**2. "Our plan has four runs: (1) an unsteered baseline, (2) activation addition at layers 16–23 using the refusal direction from Arditi et al. (arXiv:2406.11717), (3) the same with CLAS, and (4) the same with Steer2Edit. My teammates own 2 and 3; I own 4. Implement Steer2Edit in this repo, faithful to the paper and the authors’ released code. Reuse our model loading, data splits, eval prompts and metric so its results drop into the same table. Put it in separate files so it doesn’t touch theirs, document any deviation from the paper, and test it as far as possible without a GPU."**

It identified the method as Steer2Edit (Sun et al., arXiv:2602.09870), read the paper and the authors' official code ([github.com/Trustworthy-ML-Lab/Steer2Edit](https://github.com/Trustworthy-ML-Lab/Steer2Edit)), and ported it:

- `steer2edit.py`: builds per-layer attention/MLP steering vectors from the model's own responses, computes the rank-1 edit for each head and neuron (cosine importance, elastic-net λ), applies the edits, and restores the weights exactly.
- `run_steer2edit.py`: the pipeline. It uses the same eval prompts, decoding and metric as `main.py`, runs a grid over (ρ\_attn, ρ\_mlp, α) on held-out validation prompts, evaluates the selected config against the baseline, and can resume an interrupted grid.
- `STEER2EDIT.md`: usage, plus every deviation from the paper.
- `config.py`: one added line so the device can be set with `REFUSAL_DEVICE` (default still `cuda`), for a smoke test on a Mac.

It proposed the design choices (edit layers 16–23 to match the other arms, select on held-out prompts, follow the authors' code where it differs from the paper), and the author accepted them.

**3. "Write an `AI_USE.md` for this repo that meets the DSC 180A AI-use requirement. For each key prompt, document what the AI produced and how the resulting code was tested. Include only prompts that produced code, add a table separating human and AI contributions, and state clearly what has not been tested yet. Also draft a short contribution statement for the README that links to the file."**

It drafted the code sections of this file from the session history: the contribution table, the key prompts, and the testing record. It also drafted a contribution statement for the README. The author then edited the draft.

*The prompts for experiments 1–3 (4–6 below) were reconstructed afterwards from the code and README; the original chats weren't saved.*

**4. "Extract a refusal direction from Llama-3.1-8B-Instruct by difference-in-means between AdvBench harmful and Alpaca harmless prompts, at the last prompt token. Add it with a fixed coefficient α at layers 16–23 using plain PyTorch forward hooks. Sweep α, and pick the smallest value that reaches ASR ≥ 0.90 without degenerate output."**

The resulting code: `direction.py` extracts a unit direction at each of layers 16–23 in one forward pass, and `steering.py` adds α·r̂ there with forward hooks. `main.py` sweeps α from 0.25 to 2.0 and selects α = 1.0.

**5. "Implement CLAS on the same directions and layers: a learned gate c·\[h, 1\] per layer, zero-initialised, trained on MedSafetyBench train (risky) and MedQuAD (benign) prompts. Evaluate it against fixed addition on the same 200 + 200 eval prompts."**

The resulting code: `clas.py` trains one gate per layer, with a separate learning rate for its bias entry, and `steering.py` applies h ← h + (c·\[h, 1\])·r̂. After training toward a fixed refusal sentence failed, the risky class was trained toward the model's own steered refusals and the benign class toward its own unsteered answers.

**6. "Score refusals with the Arditi et al. refusal-marker list, flag degenerate outputs, and report ASR, benign over-refusal and selectivity. Add a control that swaps in a norm-matched random direction."**

The resulting code: `evaluate.py` normalises apostrophes before matching refusal markers, flags degenerate outputs, and computes ASR, over-refusal, induced ASR and selectivity. `--random-direction` reruns everything with random unit directions.

### Report (ChatGPT)

These are the prompts given to ChatGPT while writing the report, quoted as typed.

**R1. Introduction polish**

> HERE IS MY INTRODUCTION POLISH THE GRAMMAR AND WRITE BETTER: Large Language Models (LLMs) are becoming more widely used to provide information in different areas, including healthcare and medicine. However, they can also create safety risks, especially when users ask for dangerous medical instructions or harmful advice. Giving this type of information can have serius consequences, which makes it important for models to know when they shourefuse a request.Even though LLMs have safety mechnisms to prevent harmful responses, they do not always make the right decision. Sometimes they answer dangerous medical questions that should be refused, while other times they refuse harmless questions that could be answered safely. This is known as over-refusal, and it can make models less useful by preventing users from getting information they actually need.The main challange is finding a balance between safety and usefulness. Models should be able to refuse dangerous requests without refusing normal medical questions. This project focuses on improving topic-specific refusal in LLMs, particularly their ability to distinguish between dangerous medical prompts and harmless medical questions. The aim is to improve refusal behavior while still allowing models to provide useful medical information when it is safe to do so.

The introduction was drafted by the author; ChatGPT corrected the grammar and improved the wording.

**R2. Literature review feedback**

> TELL ME THE STRENGHTS AND WEAKNESS OF THIS LITERATURE REVIEW I DID A BRAIN DUMP O WHAT I KNEW:
> Finetuning is a fragile way of chnaging llm behavious ars these data sets ar every limited and can lead to various problems like racism sexism etc as shown by \cite{betley2025weird}. Steerring comes out to be a cheap way to chnage the LLm behavior on inference time as re training llms are very costly. We limit our study to refusal in autoregresive llms LLama 3.1B, we only focus on refusal because it is well studied and has a singular universal direction as shown by  \cite{arditi2024refusal}, which shows the difference of means appraoch to get the direction. There are various studies to remove done on various types of activation steering such as CLAS \cite{hsu2026contextual} \cite{turner2023steering} however we find very less attempts which focus on  doing refusal on a particular topic like medical in our case.

The content and sources came from the author; ChatGPT gave feedback on the review's strengths and weaknesses.

**R3. Method explanation and equations**

> explain activation steering and contextual linear activation steering and write it's equations and math in latex. Before writing anything refer to the sources given:
>
> ```bibtex
> @article{hsu2026contextual,
>   title={Contextual Linear Activation Steering of Language Models},
>   author={Hsu, Brandon and Beaglehole, Daniel and Radhakrishnan, Adityanarayanan and Belkin, Mikhail},
>   journal={arXiv preprint arXiv:2604.24693},
>   year={2026}
> }
>
> @article{arditi2024refusal,
>   title={Refusal in language models is mediated by a single direction},
>   author={Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
>   journal={Advances in Neural Information Processing Systems},
>   volume={37},
>   pages={136037--136083},
>   year={2024}
> }
> ```

ChatGPT explained activation steering and CLAS and wrote the equations in LaTeX, based on the two cited papers.

**R4. Writing standards**

> My report is comming together nicely, but I think has some useless info what are some of the standards of writing I should use so that the report is to the point here is the report tell me.

ChatGPT suggested writing standards for keeping the report concise and pointed out content that could be cut.

**R5. Results table**

> make the table attached
> in this latex style
>
> ```latex
> \centering
> \begin{tabular}{lll}
> \toprule
> \multicolumn{2}{c}{Part}                   \\
> \cmidrule(r){1-2}
> Name     & Description     & Size ($\mu$m) \\
> \midrule
> Dendrite & Input terminal  & $\sim$100     \\
> Axon     & Output terminal & $\sim$10      \\
> Soma     & Cell body       & up to $10^6$  \\
> \bottomrule
> \end{tabular}
> ```

ChatGPT formatted the attached table in the given LaTeX style.

## How the AI-written code was tested

The AI's environment had no GPU, and the author's laptop cannot load an 8B model. So far, testing is on a tiny randomly initialised Llama (32 layers, hidden size 64) with a locally trained tokenizer. **The code has not yet been run on Llama-3.1-8B.**

`tests/test_steer2edit_math.py` (run with `PYTHONPATH=.:tests python tests/test_steer2edit_math.py`):

1. **Span masking:** the batched, left-padded response-token means match an unpadded per-sample reference (max error \~1e-7).
2. **Edit formula:** every head's λ and k̂ and every neuron's λ match the paper's formulas, recomputed independently in a plain loop.
3. **Edit shape:** each head's ΔW has rank 1 and writes only along v̂. Logits change under the edit, and after restore they are bit-identical and a weight checksum matches.
4. **Off switch:** ρ ≤ 0 produces no edits.
5. **MLP deltas:** each edited `down_proj` column equals λ·v̂, and untouched columns are zero.

The AI ran all five checks; the author re-ran them on their Mac and they passed with the same numbers.

The AI also ran the full pipeline on the tiny model with stubbed data, to check the wiring: `--smoke`; a small full run; resuming an interrupted grid; `--direction arditi`; and `--random-direction --layers all`. It caught and fixed three problems: a test that aliased a weight tensor, high memory use from building every ΔW at once (it now streams one matrix at a time), and smoke-test output landing in the folder the real run resumes from.

`run_baseline.py` was only checked to compile. Its numbers in `outputs_baseline/` come from analysing the existing `main.py` generations, not from running it.

Experiments 1–3 were checked with the controls described in the README. A norm-matched random-direction control stays flat at baseline, so the effect comes from the refusal direction rather than from perturbing the model. Every generation is flagged for degeneracy, so collapsed output never counts as a non-refusal. An α sweep shows the full trade-off rather than a single tuned point. `data.py` asserts that every train and eval split is disjoint, and all generations are saved for hand spot-checks.

**Still to do (human):** `python run_steer2edit.py --smoke`, then the full run, on the GPU server, and spot-check the saved generations by hand.
