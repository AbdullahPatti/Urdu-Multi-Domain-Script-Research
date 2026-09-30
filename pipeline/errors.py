"""Stage 4b — error analysis for the worst-transferring shift of every matrix.

The worst shift of a matrix is the source->target pair with the largest Target Drop
for fine-tuned XLM-R (mean over seeds): the target's own model does well, the
source model does not, so the loss cannot be blamed on a hard target.

For that shift we write
  * confusion matrices (summed over the three seeds) for XLM-R trained on the
    source, XLM-R trained on the target, and the few-shot LLM with the highest
    cross-domain score in the matrix (on its evaluation sample);
  * a measured breakdown of ALL XLM-R errors on the shift. An item counts as an
    error when at least two of the three seeds misclassify it. Every category is
    computed from the data, never assigned by hand, and is reported for errors and
    for correctly classified items of the same target, so it can be read as a
    diagnostic rather than a description:
        length shift   item length (words) outside the 5th-95th percentile of the
                       source training split
        unseen vocab   more than half of the item's word types never occur in the
                       source training split
        hard in-domain the target's own XLM-R also misclassifies it (2 of 3 seeds),
                       i.e. the item is hard or mislabelled regardless of the shift
  * candidate misclassified items (short, wrong for all seeds, right in-domain)
    for the qualitative example table.

Usage:  python -m pipeline.errors
"""
from collections import Counter

import numpy as np
import pandas as pd

from pipeline.analyze import load_runs, matrix
from pipeline.common import load_split
from pipeline.config import ANALYSIS, DOMAIN_BY_KEY, TASKS

FT_MODEL = "XLM-R"
LLMS = ["Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B"]


def words(t):
    return t.split()


def seed_preds(df, fam, model, task, src, tgt):
    g = df[(df.task == task.tid) & (df.family == fam) & (df.model == model)
           & (df.src == src) & (df.tgt == tgt)].sort_values("seed")
    return g


def majority_wrong(g):
    """Boolean per item: misclassified by at least 2 of the seeds."""
    wrong = np.array([np.array(r.y_true) != np.array(r.y_pred) for r in g.itertuples()])
    return wrong.sum(0) >= 2, wrong.all(0)


def summed_confusion(g, C):
    cm = np.zeros((C, C + 1), dtype=int)      # last column = unparseable (LLM only)
    for r in g.itertuples():
        for t, p in zip(r.y_true, r.y_pred):
            cm[t, p if p >= 0 else C] += 1
    return cm


def mean_matrix(df, task, fam, model):
    g = df[(df.task == task.tid) & (df.family == fam) & (df.model == model)]
    mats = [matrix(gs, task) for _, gs in g.groupby("seed")]
    return sum(mats) / len(mats)


def main():
    df = load_runs()
    sid2key = {(t.tid, DOMAIN_BY_KEY[k].sid): k for t in TASKS for k in t.domains}   # sids repeat across scripts
    conf_rows, cat_rows, ex_rows = [], [], []
    for task in TASKS:
        C = len(task.labels)
        m = mean_matrix(df, task, "ft", FT_MODEL)
        ids = list(m.index)
        dT = {(i, j): m.loc[j, j] - m.loc[i, j] for i in ids for j in ids if i != j}
        (src, tgt), worst = max(dT.items(), key=lambda kv: kv[1])
        # best LLM in this matrix by mean cross-domain score
        llm_st = {}
        for name in LLMS:
            lm = mean_matrix(df, task, "llm", name)
            llm_st[name] = np.nanmean(lm.values[~np.eye(len(lm), dtype=bool)])
        best_llm = max(llm_st, key=llm_st.get)

        systems = {"ft_src": ("ft", FT_MODEL, src, tgt), "ft_tgt": ("ft", FT_MODEL, tgt, tgt),
                   "llm_src": ("llm", best_llm, src, tgt)}
        for tag, (fam, model, s, t) in systems.items():
            cm = summed_confusion(seed_preds(df, fam, model, task, s, t), C)
            for a in range(C):
                for b in range(C + 1):
                    conf_rows.append({"task": task.tid, "system": tag, "model": model, "src": s, "tgt": t,
                                      "true": task.label_names[a],
                                      "pred": task.label_names[b] if b < C else "unparseable",
                                      "count": int(cm[a, b])})

        # ---- measured error categories (all items of the target test split)
        g_src = seed_preds(df, "ft", FT_MODEL, task, src, tgt)
        g_tgt = seed_preds(df, "ft", FT_MODEL, task, tgt, tgt)
        err, err_all = majority_wrong(g_src)
        hard_in, _ = majority_wrong(g_tgt)
        test = load_split(sid2key[task.tid, tgt], "test", task).reset_index(drop=True)
        train = load_split(sid2key[task.tid, src], "train", task)
        assert len(test) == len(err), (task.tid, len(test), len(err))
        tr_len = train.text.map(lambda x: len(words(x)))
        lo, hi = np.percentile(tr_len, 5), np.percentile(tr_len, 95)
        vocab = set(w for t in train.text for w in words(t))
        te_len = test.text.map(lambda x: len(words(x))).values
        oov = test.text.map(lambda x: np.mean([w not in vocab for w in set(words(x))]) if words(x) else 0).values
        len_shift = (te_len < lo) | (te_len > hi)
        unseen = oov > 0.5
        y = test.y.values
        # majority prediction over seeds, for the error-direction breakdown
        preds = np.array([r.y_pred for r in g_src.itertuples()])
        maj_pred = np.array([Counter(col).most_common(1)[0][0] for col in preds.T])
        direction = Counter((task.label_names[a], task.label_names[b]) for a, b in zip(y[err], maj_pred[err]))
        top_dir, top_n = direction.most_common(1)[0]
        ok = ~err
        cat_rows.append({
            "task": task.tid, "src": src, "tgt": tgt, "dT": worst,
            "ST": m.loc[src, tgt], "TT": m.loc[tgt, tgt], "best_llm": best_llm,
            "n_test": len(test), "n_err": int(err.sum()), "err_rate": err.mean(),
            "top_dir": f"{top_dir[0]}$\\rightarrow${top_dir[1]}", "top_dir_share": top_n / max(1, err.sum()),
            "len_shift_err": len_shift[err].mean(), "len_shift_ok": len_shift[ok].mean() if ok.any() else np.nan,
            "unseen_err": unseen[err].mean(), "unseen_ok": unseen[ok].mean() if ok.any() else np.nan,
            "hard_in_err": hard_in[err].mean(), "hard_in_ok": hard_in[ok].mean() if ok.any() else np.nan,
        })

        # ---- candidate examples: wrong for every seed, right for the in-domain model
        cand = np.where(err_all & ~hard_in & (te_len <= 22))[0]
        rng = np.random.default_rng(0)
        for i in rng.permutation(cand)[:6]:
            ex_rows.append({"task": task.tid, "src": src, "tgt": tgt, "row": int(i), "text": test.text[i],
                            "gold": task.label_names[y[i]], "pred": task.label_names[maj_pred[i]],
                            "len": int(te_len[i]), "oov": round(float(oov[i]), 2)})

    pd.DataFrame(conf_rows).to_csv(ANALYSIS / "errors_confusion.csv", index=False)
    pd.DataFrame(cat_rows).to_csv(ANALYSIS / "errors_categories.csv", index=False)
    pd.DataFrame(ex_rows).to_csv(ANALYSIS / "errors_examples.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(cat_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
