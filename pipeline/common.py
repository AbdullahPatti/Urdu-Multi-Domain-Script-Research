"""Shared helpers: loading v1.1 splits, label encoding, metrics, result files."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from pipeline.config import DATA_V11, DOMAIN_BY_KEY, RUNS


def load_split(key, split, task):
    df = pd.read_csv(DATA_V11 / key / f"{split}.csv", encoding="utf-8-sig", dtype=str,
                     keep_default_na=False)
    lab = {l: i for i, l in enumerate(task.labels)}
    unknown = set(df.label) - set(lab)
    assert not unknown, f"{key}/{split}: labels {unknown} not in {task.labels}"
    df["y"] = df.label.map(lab).astype(int)
    return df


def split_pair(text):
    """QA rows are stored as 'Question: q\\nAnswer: a'."""
    q, a = text.split("\nAnswer: ", 1)
    return q.removeprefix("Question: "), a


def scores(y_true, y_pred, n_classes):
    """Macro-F1 over the true label set. Unparseable LLM outputs (-1) count as errors."""
    labels = list(range(n_classes))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels + ([-1] if -1 in set(y_pred) else [])).tolist(),
        "n": int(len(y_true)),
    }


def cell_path(family, task, model, seed, src, tgt):
    s, t = DOMAIN_BY_KEY[src].sid, DOMAIN_BY_KEY[tgt].sid
    return RUNS / family / task.tid / model / f"seed{seed}" / f"{s}__{t}.json"


def write_cell(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    os.replace(tmp, path)   # atomic: a crash never leaves a half-written cell


def to_list(a):
    return np.asarray(a).astype(int).tolist()
