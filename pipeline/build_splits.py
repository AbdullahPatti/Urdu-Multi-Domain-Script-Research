"""Stage 1 — build the corrected v1.1 splits from the public v1.0 release.

v1.0 (Hugging Face / Zenodo) = original stratified 80/20 splits (seed 42) minus
test rows whose exact text occurs in the same domain's train split (4,196 rows).

v1.1 applies four further corrections, every one logged per domain:
  1. drop domains that duplicate another domain (config.DROPPED_V11)
  1b. QA pair validation is rebuilt from the question/answer tables with a
     question-grouped split and within-split distractors (rebuild_qa)
  2. exact de-duplication inside each split, after whitespace normalisation
  3. cross-domain overlap: a text shared by two domains of one transfer matrix
     is removed from the larger domain (both splits), so no ST cell ever tests
     on text its model was trained on
  4. near-duplicates (MinHash over character 5-grams, verified Jaccard >= 0.8),
     grouped into clusters by union-find:
       a. inside a domain, if more than 10% of test rows fall in a cluster that
          also has train rows (typical of template-expanded synthetic data), the
          domain is re-split 80/20 so that whole clusters stay on one side
          (StratifiedGroupKFold, seed 42); otherwise the leaking test rows are removed
       b. near-duplicates shared by two domains of one matrix are removed from the
          larger domain

Usage:  python -m pipeline.build_splits --v10 <dir with the v1.0 release>
"""
import argparse
import json
import shutil
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedGroupKFold

from pipeline.config import (ANALYSIS, DATA_V10, DATA_V11, DOMAINS, DOMAIN_BY_KEY,
                             DROPPED_V11, RAW_SPLITS, TASKS)

JACCARD = 0.8
RESPLIT_RATE = 0.10   # re-split a domain cluster-aware if >10% of its test rows leak
NUM_PERM = 128
SHINGLE = 5


def norm(text: str) -> str:
    return " ".join(str(text).split())


def read_split(base, key, split):
    df = pd.read_csv(base / key / f"{split}.csv", encoding="utf-8-sig", dtype=str,
                     keep_default_na=False)
    return df[["text", "label"]]


def verify_v10_against_raw():
    """Re-derive v1.0 from the legacy splits (non-QA domains) and assert equality."""
    for d in DOMAINS:
        if d.collection == "QA":   # QA pairs are built by pairing; checked on overlap only
            tr, te = read_split(DATA_V10, d.key, "train"), read_split(DATA_V10, d.key, "test")
            assert not set(tr.text) & set(te.text), d.key
            continue
        raw_tr, raw_te = read_split(RAW_SPLITS, d.key, "train"), read_split(RAW_SPLITS, d.key, "test")
        rebuilt = raw_te[~raw_te.text.isin(set(raw_tr.text))].reset_index(drop=True)
        rel_te = read_split(DATA_V10, d.key, "test")
        assert raw_tr.equals(read_split(DATA_V10, d.key, "train")), f"train differs: {d.key}"
        assert rebuilt.equals(rel_te), f"test differs: {d.key}"
    print("v1.0 release re-derived exactly from legacy splits (non-QA) and QA overlap = 0")


def minhash(text):
    m = MinHash(num_perm=NUM_PERM, seed=1)
    t = norm(text)
    grams = {t[i:i + SHINGLE] for i in range(max(1, len(t) - SHINGLE + 1))}
    for g in grams:
        m.update(g.encode("utf-8"))
    return m, grams


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def rebuild_qa(d, test_frac=0.2, seed=42):
    """Rebuild a QA pair-validation domain from its question/answer table so that no
    question (or near-identical question, or distinctive answer of >= 4 words) occurs
    in both splits, then create pairs inside each split: every question with its own
    answer (valid) and with an answer from a different group (invalid)."""
    raw = pd.concat([read_split(RAW_SPLITS, d.key, s) for s in ("train", "test")], ignore_index=True)
    raw = raw.rename(columns={"text": "q", "label": "a"})
    raw["qn"], raw["an"] = raw.q.map(norm), raw.a.map(norm)
    raw = raw[(raw.qn != "") & (raw.an != "")].drop_duplicates(["qn", "an"]).reset_index(drop=True)
    comp, _ = near_dup_clusters(raw.qn.tolist())
    parent = list(range(len(raw)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    distinctive = raw.an.where(raw.an.str.split().str.len() >= 4)
    for key in (pd.Series(comp), raw.qn, distinctive):
        first = {}
        for i, v in enumerate(key):
            if pd.isna(v):
                continue
            if v in first:
                a, b = find(i), find(first[v])
                parent[max(a, b)] = min(a, b)
            else:
                first[v] = i
    raw["group"] = [find(i) for i in range(len(raw))]
    sizes = raw.group.value_counts()
    rng = np.random.default_rng(seed)
    order = list(sizes.index[1:])
    rng.shuffle(order)
    test_groups, n_test = set(), 0
    for g in order:                       # largest group always stays in train
        if n_test >= test_frac * len(raw):
            break
        test_groups.add(g)
        n_test += sizes[g]
    out = {}
    for split, part in (("train", raw[~raw.group.isin(test_groups)]), ("test", raw[raw.group.isin(test_groups)])):
        part = part.reset_index(drop=True)
        neg = []
        for i in range(len(part)):
            while True:
                j = int(rng.integers(len(part)))
                if part.group[j] != part.group[i] and part.an[j] != part.an[i]:
                    break
            neg.append(part.a[j])
        pairs = pd.concat([
            pd.DataFrame({"text": "Question: " + part.q + "\nAnswer: " + part.a, "label": "1"}),
            pd.DataFrame({"text": "Question: " + part.q + "\nAnswer: " + pd.Series(neg), "label": "0"}),
        ], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
        out[split] = pairs
    info = {"qa_rows": len(raw), "qa_groups": int(len(sizes)), "qa_largest_group": int(sizes.iloc[0]),
            "qa_test_questions": int(n_test)}
    return out["train"], out["test"], info


def near_dup_clusters(texts, only_between=None):
    """Union-find over pairs with verified char-5-gram Jaccard >= JACCARD (MinHash LSH
    candidates). With `only_between`, only pairs whose group labels differ are kept.
    Returns (component id per text, list of (i, j, jaccard))."""
    lsh = MinHashLSH(threshold=JACCARD, num_perm=NUM_PERM)
    hs, gs = [], []
    for i, t in enumerate(texts):
        h, g = minhash(t)
        hs.append(h); gs.append(g)
        lsh.insert(str(i), h)
    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    pairs = []
    for i in range(len(texts)):
        # sorted: LSH returns a set, whose order depends on Python's per-process hash seed
        for j in sorted(int(c) for c in lsh.query(hs[i])):
            if j <= i or (only_between and only_between[i] == only_between[j]):
                continue
            jac = jaccard(gs[i], gs[j])
            if jac >= JACCARD:
                pairs.append((i, j, round(jac, 3)))
                a, b = find(i), find(j)
                parent[max(a, b)] = min(a, b)   # root = smallest index: order-independent ids
    return [find(i) for i in range(len(texts))], pairs


def build(v10_dir):
    if v10_dir:
        src = __import__("pathlib").Path(v10_dir)
        if DATA_V10.exists():
            shutil.rmtree(DATA_V10)
        for d in DOMAINS:
            (DATA_V10 / d.key).mkdir(parents=True, exist_ok=True)
            for split in ("train", "test"):
                shutil.copy(src / d.key / f"{split}.csv", DATA_V10 / d.key / f"{split}.csv")
    verify_v10_against_raw()

    log = defaultdict(lambda: defaultdict(int))
    data = {}
    for d in DOMAINS:
        tr, te = read_split(DATA_V10, d.key, "train"), read_split(DATA_V10, d.key, "test")
        log[d.key]["v1.0_train"], log[d.key]["v1.0_test"] = len(tr), len(te)
        if d.key in DROPPED_V11:
            log[d.key]["dropped_domain"] = len(tr) + len(te)
            continue
        if d.collection == "QA":
            # (1b) question-grouped rebuild: v1.0 split QA *pairs*, so 28-84% of test
            # questions also occurred in train (memorisable question->answer facts)
            tr, te, info = rebuild_qa(d)
            log[d.key].update(info)
        for df in (tr, te):
            df["norm"] = df.text.map(norm)
        # (2) exact duplicates inside a split, and test rows equal to train after normalisation
        n0 = len(tr); tr = tr.drop_duplicates("norm"); log[d.key]["within_train_dup"] = n0 - len(tr)
        n0 = len(te); te = te.drop_duplicates("norm"); log[d.key]["within_test_dup"] = n0 - len(te)
        n0 = len(te); te = te[~te.norm.isin(set(tr.norm))]; log[d.key]["test_in_train_normalised"] = n0 - len(te)
        data[d.key] = {"train": tr.reset_index(drop=True), "test": te.reset_index(drop=True)}

    # (3) exact cross-domain overlap inside each transfer matrix
    for task in TASKS:
        for a, b in combinations(task.domains, 2):
            sa = set(data[a]["train"].norm) | set(data[a]["test"].norm)
            sb = set(data[b]["train"].norm) | set(data[b]["test"].norm)
            shared = sa & sb
            if not shared:
                continue
            loser = a if len(sa) > len(sb) else b
            for split in ("train", "test"):
                df = data[loser][split]
                keep = ~df.norm.isin(shared)
                log[loser][f"cross_domain_exact_{split}"] += int((~keep).sum())
                data[loser][split] = df[keep].reset_index(drop=True)
            partner = a if loser == b else b
            log[loser].setdefault("overlap_partners", "")
            log[loser]["overlap_partners"] += f"{DOMAIN_BY_KEY[partner].name} ({len(shared)}); "

    # (4a) near-duplicates inside each domain (train vs test)
    near_log = []
    for d in DOMAINS:
        if d.key not in data:
            continue
        tr, te = data[d.key]["train"], data[d.key]["test"]
        both = pd.concat([tr.assign(split="train"), te.assign(split="test")], ignore_index=True)
        comp, pairs = near_dup_clusters(both.norm.tolist())
        both["cluster"] = comp
        train_clusters = set(both.loc[both.split == "train", "cluster"])
        leaked = (both.split == "test") & both.cluster.isin(train_clusters)
        rate = leaked.sum() / max(1, len(te))
        L = log[d.key]
        L["near_dup_test_rate"] = round(float(rate), 4)
        L["near_dup_clusters_multi"] = int((both.cluster.value_counts() > 1).sum())
        for i, j, jac in pairs:
            near_log.append({"domain": d.key, "jaccard": jac, "a_split": both.split[i], "b_split": both.split[j],
                             "a": both.norm[i][:200], "b": both.norm[j][:200]})
        if rate > RESPLIT_RATE and d.collection != "QA":   # QA is already question-grouped (1b)
            # cluster-aware re-split: whole near-duplicate clusters go to one side (80/20, stratified)
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            tr_idx, te_idx = next(sgkf.split(both, both.label, groups=both.cluster))
            data[d.key] = {"train": both.iloc[tr_idx].reset_index(drop=True)[["text", "label", "norm"]],
                           "test": both.iloc[te_idx].reset_index(drop=True)[["text", "label", "norm"]]}
            L["cluster_resplit"] = 1
        else:
            data[d.key]["test"] = both[(both.split == "test") & ~leaked][["text", "label", "norm"]].reset_index(drop=True)
            L["near_dup_test_removed"] = int(leaked.sum())

    # (4b) near-duplicates across domains of one transfer matrix -> drop from the larger domain
    for task in TASKS:
        rows = [(k, sp, i, t) for k in task.domains for sp in ("train", "test")
                for i, t in enumerate(data[k][sp].norm)]
        _, pairs = near_dup_clusters([r[3] for r in rows], only_between=[r[0] for r in rows])
        sizes = {k: len(data[k]["train"]) + len(data[k]["test"]) for k in task.domains}
        drop = defaultdict(set)
        for i, j, jac in pairs:
            a, b = rows[i], rows[j]
            victim = a if sizes[a[0]] > sizes[b[0]] else b
            if victim[2] not in drop[(victim[0], victim[1])]:
                drop[(victim[0], victim[1])].add(victim[2])
                log[victim[0]][f"near_dup_cross_domain_{victim[1]}"] += 1
        for (k, sp), idx in drop.items():
            data[k][sp] = data[k][sp].drop(index=sorted(idx)).reset_index(drop=True)

    # write v1.1
    if DATA_V11.exists():
        shutil.rmtree(DATA_V11)
    rows = []
    for d in DOMAINS:
        L = log[d.key]
        row = {"key": d.key, "script": d.script, "collection": d.collection, "domain": d.name,
               "letter": d.letter, "origin": d.origin,
               **{k: v for k, v in L.items()}}
        if d.key in data:
            (DATA_V11 / d.key).mkdir(parents=True, exist_ok=True)
            for split in ("train", "test"):
                df = data[d.key][split][["text", "label"]]
                df.to_csv(DATA_V11 / d.key / f"{split}.csv", index=False, encoding="utf-8-sig")
                row[f"v1.1_{split}"] = len(df)
                row[f"v1.1_{split}_dist"] = json.dumps(df.label.value_counts().sort_index().to_dict())
        else:
            row["dropped_reason"] = DROPPED_V11[d.key]
        rows.append(row)
    ANALYSIS.mkdir(exist_ok=True)
    audit = pd.DataFrame(rows).fillna(0)
    audit.to_csv(ANALYSIS / "data_audit_v11.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(near_log).to_csv(ANALYSIS / "near_duplicates_v11.csv", index=False, encoding="utf-8-sig")
    kept = audit[audit.get("v1.1_train", 0) != 0]
    print(f"v1.1: {len(kept)} domains, {int(kept['v1.1_train'].sum())} train + "
          f"{int(kept['v1.1_test'].sum())} test = {int(kept['v1.1_train'].sum() + kept['v1.1_test'].sum())}")
    cols = ["collection", "letter", "origin", "v1.0_train", "v1.0_test", "near_dup_test_rate",
            "cluster_resplit", "v1.1_train", "v1.1_test"]
    print(audit.reindex(columns=cols).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10", help="directory holding the downloaded v1.0 release (copied into data/v1.0)")
    build(ap.parse_args().v10)
