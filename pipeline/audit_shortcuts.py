"""Audit — single-phrase shortcuts that reveal where an item came from.

For every domain (train + test pooled) and every word n-gram (n = 1..3) we compute
its document frequency in each class. The shortcut score of the domain is the
largest one-vs-rest difference in document frequency: a score of 0.9 means some
phrase occurs in 90 percentage points more items of one class than of the others.
High scores are expected for content words (a slur in hate speech, "bakwas" in
negative reviews); they are a problem when the phrase is source boilerplate (an
outlet dateline, a "web desk" byline), because then the label can be read off the
origin of the item. The top phrases are written out for inspection.

Included domains are read from the benchmark splits; excluded candidates of the
same matrices are read from the original v1.0 files, which contain all candidates.

Usage:  python -m pipeline.audit_shortcuts   -> analysis/shortcut_audit.csv
"""
from collections import Counter

import pandas as pd

from pipeline.common import split_pair
from pipeline.config import ANALYSIS, DATA_V10, DATA_V11, DOMAIN_BY_KEY, DROPPED_V11, TASKS

MIN_DF = 20   # ignore phrases seen in fewer than 20 items of the domain


def ngrams(text, n_max=3):
    w = text.split()
    return {" ".join(w[i:i + n]) for n in range(1, n_max + 1) for i in range(len(w) - n + 1)}


def load(root, key, task):
    df = pd.concat([pd.read_csv(root / key / f"{s}.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
                    for s in ("train", "test")], ignore_index=True)
    lab = {l: i for i, l in enumerate(task.labels)}
    df = df[df.label.isin(lab)].copy()
    df["y"] = df.label.map(lab)
    return df


def audit_domain(df, task, top=5):
    if task.pair_input:   # QA: audit the answers, where a source marker would sit
        df = df.assign(text=df.text.map(lambda t: split_pair(t)[1]))
    grams = [ngrams(t) for t in df.text]
    classes = sorted(df.y.unique())
    n_c = Counter(df.y)
    df_c = {c: Counter() for c in classes}
    for g, y in zip(grams, df.y):
        df_c[y].update(g)
    total = Counter()
    for c in classes:
        total.update(df_c[c])
    rows = []
    for gram, tot in total.items():
        if tot < MIN_DF:
            continue
        for c in classes:
            p_in = df_c[c][gram] / n_c[c]
            p_out = (tot - df_c[c][gram]) / (len(df) - n_c[c])
            rows.append((p_in - p_out, gram, task.label_names[c], p_in, p_out))
    rows.sort(reverse=True)
    return rows[:top]


def main():
    out = []
    for t in TASKS:
        colls = {(DOMAIN_BY_KEY[k].script, DOMAIN_BY_KEY[k].collection) for k in t.domains}
        excluded = [k for k in DROPPED_V11 if (DOMAIN_BY_KEY[k].script, DOMAIN_BY_KEY[k].collection) in colls
                    and (DATA_V10 / k / "train.csv").exists()]
        for k, root, inc in [(k, DATA_V11, True) for k in t.domains] + [(k, DATA_V10, False) for k in excluded]:
            for rank, (diff, gram, lab, p_in, p_out) in enumerate(audit_domain(load(root, k, t), t), 1):
                out.append({"task": t.tid, "domain": DOMAIN_BY_KEY[k].sid, "included": inc,
                            "rank": rank, "phrase": gram, "class": lab,
                            "df_in_class": round(p_in, 3), "df_other": round(p_out, 3), "diff": round(diff, 3)})
    res = pd.DataFrame(out)
    res.to_csv(ANALYSIS / "shortcut_audit.csv", index=False, encoding="utf-8-sig")
    print(res[res["rank"] == 1].to_string(index=False))


if __name__ == "__main__":
    main()
