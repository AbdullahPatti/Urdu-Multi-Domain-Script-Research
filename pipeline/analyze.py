"""Stage 4 — turn runs/ into every table, statistic and figure the paper reports.

Definitions (Calderon et al., 2024), for one model and one seed, with F(i->j) the
macro-F1 of the model trained/prompted on domain i and tested on domain j:
    SS_i = F(i->i)      TT_j = F(j->j)      ST_ij = F(i->j), i != j
    dS_ij = SS_i - ST_ij   (Source Drop)    dT_ij = TT_j - ST_ij   (Target Drop)
Over a complete n x n matrix, mean(dS) == mean(dT) identically, because every
domain is a source n-1 times and a target n-1 times. Target Drop is therefore
reported per shift and per target domain, through worst cases, the share of
shifts in which each drop is positive, and the share of "harder-target" shifts
(dS > 0 but dT <= 0: the loss is explained by moving to a harder domain).

Aggregates are computed per seed and reported as mean +/- SD across seeds.
Degeneracy guard: %Rob is suppressed (dagger) when SS does not exceed the
uniform-random baseline by DEGEN_MARGIN, or when %Rob >= 100.

Usage:  python -m pipeline.analyze
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from pipeline.common import load_split
from pipeline.config import ANALYSIS, DOMAIN_BY_KEY, RUNS, TASKS, TASK_BY_ID

DEGEN_MARGIN = 0.05
N_BOOT = 5000
FT_ORDER = ["XLM-R", "mBERT"]
LLM_ORDER = ["Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B"]


# ----------------------------------------------------------------------------- loading
def load_runs():
    cells = []
    for p in RUNS.glob("*/*/*/seed*/*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        r.pop("raw", None)
        cells.append(r)
    df = pd.DataFrame(cells)
    # cells of excluded domains (config.DROPPED_V11) may still exist on disk from earlier
    # runs; only shifts between domains of the current transfer matrices are analysed
    active = {k for t in TASKS for k in t.domains}
    df = df[df.source.isin(active) & df.target.isin(active)].reset_index(drop=True)
    df["src"] = df.source.map(lambda k: DOMAIN_BY_KEY[k].sid)
    df["tgt"] = df.target.map(lambda k: DOMAIN_BY_KEY[k].sid)
    return df


def matrix(cells, task):
    ids = [DOMAIN_BY_KEY[k].sid for k in task.domains]
    m = pd.DataFrame(np.nan, index=ids, columns=ids)
    for r in cells.itertuples():
        m.loc[r.src, r.tgt] = r.macro_f1
    return m


# ----------------------------------------------------------------------------- baselines
def baselines(task):
    """Majority-class and expected uniform-random macro-F1, averaged over target domains."""
    maj, rnd = [], []
    C = len(task.labels)
    for k in task.domains:
        y = load_split(k, "test", task).y.values
        mc = np.bincount(y, minlength=C).argmax()
        maj.append(f1_score(y, np.full_like(y, mc), labels=range(C), average="macro", zero_division=0))
        p = np.bincount(y, minlength=C) / len(y)
        rnd.append(np.mean([2 * pc * (1 / C) / (pc + 1 / C) if pc > 0 else 0 for pc in p]))
    return float(np.mean(maj)), float(np.mean(rnd))


def length_matrix(task):
    """Shortcut baseline: logistic regression on log character and word length only,
    trained on each source and applied to every target (cf. the length confound in
    Urdu fake news, Haroon 2026). A domain where this scores well has a label that is
    partly predictable from length alone."""
    from sklearn.linear_model import LogisticRegression

    def feats(df):
        return np.c_[np.log1p(df.text.str.len()), np.log1p(df.text.str.split().str.len())]
    ids = [DOMAIN_BY_KEY[k].sid for k in task.domains]
    tests = {k: load_split(k, "test", task) for k in task.domains}
    m = pd.DataFrame(np.nan, index=ids, columns=ids)
    for s in task.domains:
        tr = load_split(s, "train", task)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(feats(tr), tr.y)
        for t in task.domains:
            te = tests[t]
            m.loc[DOMAIN_BY_KEY[s].sid, DOMAIN_BY_KEY[t].sid] = f1_score(
                te.y, clf.predict(feats(te)), labels=range(len(task.labels)), average="macro", zero_division=0)
    return m


# ----------------------------------------------------------------------------- per-matrix stats
def matrix_stats(m):
    ids = list(m.index)
    ss = np.array([m.loc[i, i] for i in ids])
    pairs = [(i, j) for i in ids for j in ids if i != j]
    st = np.array([m.loc[i, j] for i, j in pairs])
    dS = np.array([m.loc[i, i] - m.loc[i, j] for i, j in pairs])
    dT = np.array([m.loc[j, j] - m.loc[i, j] for i, j in pairs])
    SS, ST = ss.mean(), st.mean()
    return {
        "SS": SS, "ST": ST, "dS": dS.mean(), "WSD": dS.max(), "WTD": dT.max(),
        "pos_dS": (dS > 0).mean(), "pos_dT": (dT > 0).mean(),
        "harder_target": ((dS > 0) & (dT <= 0)).mean(),
        "rob": ST / SS if SS > 0 else np.nan,
    }


def per_domain(m):
    rows = []
    for d in m.index:
        others = [x for x in m.index if x != d]
        rows.append({"domain": d, "SS=TT": m.loc[d, d],
                     "ST_out": m.loc[d, others].mean(), "dS_source": (m.loc[d, d] - m.loc[d, others]).mean(),
                     "ST_in": m.loc[others, d].mean(), "dT_target": (m.loc[d, d] - m.loc[others, d]).mean()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main tables
def summarize(df, task, restrict=None):
    """Mean +/- SD over seeds of matrix_stats, per model. `restrict` = subset of domain sids."""
    out = []
    for (fam, model), g in df[df.task == task.tid].groupby(["family", "model"]):
        per_seed = []
        for seed, gs in g.groupby("seed"):
            m = matrix(gs, task)
            if restrict:
                m = m.loc[restrict, restrict]
            if m.isna().values.any():
                continue
            per_seed.append(matrix_stats(m))
        if not per_seed:
            continue
        s = pd.DataFrame(per_seed)
        row = {"task": task.tid, "family": fam, "model": model, "n_seeds": len(s)}
        for c in s.columns:
            row[c], row[c + "_sd"] = s[c].mean(), s[c].std(ddof=1) if len(s) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out)


def apply_guard(tab, base_rand):
    tab = tab.copy()
    tab["degenerate"] = (tab.SS < base_rand + DEGEN_MARGIN) | (tab.rob >= 1.0)
    return tab


# ----------------------------------------------------------------------------- bootstrap
def paired_bootstrap_st(df, task, fam_a, model_a, fam_b, model_b, rng):
    """Bootstrap over test items (resampled per target, shared by both systems) of the
    difference in mean cross-domain ST, averaged over seeds. LLM cells define the item
    subset; fine-tuned predictions are restricted to the same rows."""
    sub = df[df.task == task.tid]
    llm_rows = {}
    for r in sub[sub.family == "llm"].itertuples():
        llm_rows.setdefault(r.tgt, r.rows)

    def preds(fam, model):
        d = defaultdict(dict)   # (src,tgt) -> seed -> (y_true, y_pred) on shared rows
        for r in sub[(sub.family == fam) & (sub.model == model)].itertuples():
            if r.src == r.tgt:
                continue
            yt, yp = np.array(r.y_true), np.array(r.y_pred)
            if fam == "ft" and r.tgt in llm_rows:
                idx = np.array(llm_rows[r.tgt]); yt, yp = yt[idx], yp[idx]
            d[(r.src, r.tgt)][r.seed] = (yt, yp)
        return d

    A, B = preds(fam_a, model_a), preds(fam_b, model_b)
    keys = sorted(set(A) & set(B))
    if not keys:
        return None
    C = len(task.labels)
    n_items = {t: len(next(iter(A[(s, t)].values()))[0]) for s, t in keys}
    # one resample matrix per target, shared by both systems (paired); row 0 = original
    samp = {t: np.vstack([np.arange(n), rng.integers(0, n, (N_BOOT, n))]) for t, n in n_items.items()}

    def macro_f1_boot(yt, yp, idx):
        """Macro-F1 for every row of the resample matrix idx (vectorised)."""
        f1 = np.zeros(idx.shape[0])
        for c in range(C):
            tp = ((yt == c) & (yp == c))[idx].sum(1)
            fp = ((yt != c) & (yp == c))[idx].sum(1)
            fn = ((yt == c) & (yp != c))[idx].sum(1)
            den = 2 * tp + fp + fn
            f1 += np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
        return f1 / C

    def st(P):
        return np.mean([np.mean([macro_f1_boot(yt, yp, samp[t]) for yt, yp in P[(s, t)].values()], axis=0)
                        for s, t in keys], axis=0)

    d = st(A) - st(B)
    obs, diffs = d[0], d[1:]
    # two-sided bootstrap p with the (k+1)/(N+1) correction, so p is never exactly 0
    k = min((diffs <= 0).sum(), (diffs >= 0).sum())
    p = min(1.0, 2 * (k + 1) / (N_BOOT + 1))
    return {"task": task.tid, "a": model_a, "b": model_b, "diff_ST": obs,
            "ci_lo": np.percentile(diffs, 2.5), "ci_hi": np.percentile(diffs, 97.5), "p": p}


def holm(p):
    """Holm-Bonferroni adjusted p-values over one family of tests."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ANALYSIS.mkdir(exist_ok=True)
    df = load_runs()
    rng = np.random.default_rng(0)
    main_rows, organic_rows, dom_rows, parse_rows, boot_rows, base_rows = [], [], [], [], [], []
    for task in TASKS:
        if not (df.task == task.tid).any():
            continue
        bm, br = baselines(task)
        base_rows.append({"task": task.tid, "majority": bm, "random": br})
        lm = length_matrix(task)
        lm.to_csv(ANALYSIS / f"matrix_{task.tid}_Length-only.csv")
        ls = matrix_stats(lm)
        tab = apply_guard(pd.concat([summarize(df, task), pd.DataFrame([{
            "task": task.tid, "family": "baseline", "model": "Length-only", "n_seeds": 1,
            **ls, **{c + "_sd": 0.0 for c in ls}}])], ignore_index=True), br)
        main_rows.append(tab)
        organic = [DOMAIN_BY_KEY[k].sid for k in task.domains if DOMAIN_BY_KEY[k].origin == "organic"]
        if len(organic) >= 2:
            ot = apply_guard(summarize(df, task, restrict=organic), br)
            ot["domains"] = ",".join(organic)
            organic_rows.append(ot)
        for (fam, model), g in df[df.task == task.tid].groupby(["family", "model"]):
            mats = [matrix(gs, task) for _, gs in g.groupby("seed")]
            mean_m = sum(mats) / len(mats)
            mean_m.to_csv(ANALYSIS / f"matrix_{task.tid}_{model}.csv")
            pdm = per_domain(mean_m); pdm.insert(0, "model", model); pdm.insert(0, "task", task.tid)
            dom_rows.append(pdm)
            if fam == "llm":
                parse_rows.append({"task": task.tid, "model": model,
                                   "parse_fail_rate": g.parse_failures.sum() / g.n.sum()})
        for a in FT_ORDER:
            for b in LLM_ORDER:
                r = paired_bootstrap_st(df, task, "ft", a, "llm", b, rng)
                if r:
                    boot_rows.append(r)
    pd.concat(main_rows).to_csv(ANALYSIS / "table_main.csv", index=False)
    if organic_rows:
        pd.concat(organic_rows).to_csv(ANALYSIS / "table_organic_only.csv", index=False)
    pd.concat(dom_rows).to_csv(ANALYSIS / "table_per_domain.csv", index=False)
    pd.DataFrame(base_rows).to_csv(ANALYSIS / "baselines.csv", index=False)
    if parse_rows:
        pd.DataFrame(parse_rows).to_csv(ANALYSIS / "parse_failures.csv", index=False)
    if boot_rows:
        bt = pd.DataFrame(boot_rows)
        bt["p_holm"] = holm(bt.p)   # one family: all fine-tuned vs LLM comparisons
        bt.to_csv(ANALYSIS / "bootstrap_ft_vs_llm.csv", index=False)
    print(pd.concat(main_rows)[["task", "model", "n_seeds", "SS", "SS_sd", "ST", "ST_sd", "dS", "WSD",
                                "WTD", "pos_dS", "pos_dT", "harder_target", "rob", "degenerate"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
