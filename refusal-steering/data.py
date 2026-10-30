"""Loaders and splits for the four data sources.

Each source is used for exactly ONE purpose:
  AdvBench  -> direction extraction (harmful)     | never evaluated on
  Alpaca    -> direction extraction (harmless)    | never evaluated on
  MedSafetyBench -> risky eval (test split) + CLAS risky training (train split)
  MedQuAD        -> benign eval + CLAS benign training (disjoint question groups)

The medical sets are never used for direction extraction.
"""

from __future__ import annotations

import io
import random
import re
import urllib.request
from typing import Dict, List, Tuple

import pandas as pd
from datasets import load_dataset

import config


# ------------------------------------------------------------------ general (direction)
def load_advbench(n: int, seed: int = config.SEED) -> List[str]:
    """AdvBench harmful behaviours. CSV columns: goal,target (520 rows). Returns `goal`."""
    raw = urllib.request.urlopen(config.ADVBENCH_URL, timeout=60).read().decode("utf-8")
    goals = pd.read_csv(io.StringIO(raw))["goal"].astype(str).str.strip().tolist()
    goals = sorted(set(goals))  # AdvBench contains near-duplicate paraphrases
    return random.Random(seed).sample(goals, min(n, len(goals)))


def load_alpaca(n: int, seed: int = config.SEED) -> List[str]:
    """Alpaca instructions. Only rows with an empty `input` field, so each item is
    self-contained and comparable in form to an AdvBench goal."""
    ds = load_dataset("tatsu-lab/alpaca")["train"]
    instrs = [
        i.strip() for i, inp in zip(ds["instruction"], ds["input"]) if not str(inp).strip()
    ]
    instrs = sorted(set(instrs))
    return random.Random(seed).sample(instrs, min(n, len(instrs)))


# ------------------------------------------------------------------ risky (MedSafetyBench)
def _load_medsafety_split(split: str) -> pd.DataFrame:
    """Read one MedSafetyBench split. Files are
    datasets/{split}/{gpt4,llama2}/med_safety_demonstrations_category_{1..9}.csv
    with columns: <unnamed index>, harmful_medical_request, safe_response."""
    frames = []
    for generator in ("gpt4", "llama2"):
        for category in range(1, 10):
            path = (
                config.MEDSAFETY_DIR
                / split
                / generator
                / f"med_safety_demonstrations_category_{category}.csv"
            )
            df = pd.read_csv(path)
            df["generator"] = generator
            df["category"] = category
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    assert "harmful_medical_request" in out.columns, out.columns.tolist()
    return out


def load_medsafety(split: str, n: int, seed: int = config.SEED) -> List[str]:
    """`split` is "train" (CLAS risky class) or "test" (risky eval). Disjoint upstream."""
    df = _load_medsafety_split(split)
    prompts = df["harmful_medical_request"].astype(str).str.strip().tolist()
    prompts = sorted(set(prompts))
    return random.Random(seed).sample(prompts, min(n, len(prompts)))


# ------------------------------------------------------------------ benign (MedQuAD)
def _norm_group(focus: str, document_id: str) -> str:
    """Group key for the MedQuAD split: the normalised condition name, falling back to
    document_id. Grouping on document_id alone leaks, because the same condition appears
    under different document_ids across sources (GHR / GARD / NIDDK all cover diabetes)."""
    f = re.sub(r"[^a-z0-9]+", " ", str(focus).lower()).strip()
    return f if f and f != "none" else f"doc::{document_id}"


def load_medquad_groups(seed: int = config.SEED) -> Tuple[List[Dict], List[Dict]]:
    """Return (train_rows, eval_rows) with DISJOINT question groups.

    MedQuAD ships 47441 rows but 31034 have empty answers (removed for copyright), so
    those are dropped first -> 16407 usable. Questions are then grouped by condition and
    the groups, not the rows, are split; we draw ONE question per group for the eval arm
    so the eval prompts are genuinely independent rather than near-duplicates.
    """
    ds = load_dataset("lavita/MedQuAD")["train"]
    rows = [
        {
            "question": str(q).strip(),
            "answer": str(a).strip(),
            "group": _norm_group(f, d),
            "question_type": str(t),
        }
        for q, a, f, d, t in zip(
            ds["question"], ds["answer"], ds["question_focus"],
            ds["document_id"], ds["question_type"],
        )
        if a is not None and str(a).strip() and q is not None and str(q).strip()
    ]
    assert all(r["answer"] for r in rows), "empty answers must already be dropped"

    by_group: Dict[str, List[Dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)

    groups = sorted(by_group)
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_eval_groups = len(groups) // 2
    eval_groups, train_groups = groups[:n_eval_groups], groups[n_eval_groups:]

    eval_rows = [rng.choice(by_group[g]) for g in eval_groups]   # one question per group
    train_rows = [rng.choice(by_group[g]) for g in train_groups]
    rng.shuffle(eval_rows)
    rng.shuffle(train_rows)
    return train_rows, eval_rows


def truncate_answer(answer: str, max_words: int = 60) -> str:
    """Keep CLAS benign targets short, cutting at a sentence boundary where possible."""
    words = answer.split()
    if len(words) <= max_words:
        return answer.strip()
    clipped = " ".join(words[:max_words])
    cut = max(clipped.rfind(". "), clipped.rfind("? "), clipped.rfind("! "))
    return (clipped[: cut + 1] if cut > 40 else clipped).strip()


# ------------------------------------------------------------------ assembly
def build_all(seed: int = config.SEED, n_eval: int = None, n_clas: int = None,
              n_extract: int = None) -> Dict[str, object]:
    """Build every split and assert the role/disjointness invariants."""
    n_eval = config.N_EVAL if n_eval is None else n_eval
    n_clas = config.CLAS_N_PER_CLASS if n_clas is None else n_clas
    n_extract = config.N_EXTRACT if n_extract is None else n_extract

    harmful = load_advbench(n_extract, seed)
    harmless = load_alpaca(n_extract, seed)

    risky_eval = load_medsafety("test", n_eval, seed)
    risky_train = load_medsafety("train", n_clas, seed)

    mq_train, mq_eval = load_medquad_groups(seed)
    benign_eval = [r["question"] for r in mq_eval[:n_eval]]
    benign_train_rows = mq_train[:n_clas]

    # --- invariants, asserted rather than assumed
    assert not (set(risky_eval) & set(risky_train)), "MedSafetyBench train/test overlap"
    assert not (set(benign_eval) & {r["question"] for r in benign_train_rows}), \
        "MedQuAD train/eval question overlap"
    assert not ({r["group"] for r in mq_train} & {r["group"] for r in mq_eval}), \
        "MedQuAD group leakage across the split"
    medical = set(risky_eval) | set(risky_train) | set(benign_eval)
    assert not (medical & (set(harmful) | set(harmless))), \
        "medical data must never reach direction extraction"

    return {
        "harmful": harmful,
        "harmless": harmless,
        "risky_eval": risky_eval,
        "risky_train": risky_train,
        "benign_eval": benign_eval,
        "benign_train_rows": benign_train_rows,
    }
