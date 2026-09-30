"""Empirical check of how much of each domain looks LLM-generated.

The synthetic percentages in the v1.0 dataset card were not measured, so this
script measures signals on the v1.0 data (all 33 domains, train+test):
  template_rate  share of rows with a near-duplicate variant (char-5-gram Jaccard
                 >= 0.8) inside the same domain -- template expansion leaves clusters
  distinct2      distinct word bigrams / all bigrams in a fixed 500-row sample
                 (size-controlled lexical diversity)
  len_cv         coefficient of variation of length in words (generated text is uniform)
  seed_match     share of rows that (near-)duplicate a row of another domain of the
                 same script, i.e. copied or lightly edited seed material
Writes analysis/synthetic_audit.csv.
"""
import numpy as np
import pandas as pd
from datasketch import MinHashLSH

from pipeline.build_splits import JACCARD, NUM_PERM, jaccard, minhash, near_dup_clusters, norm
from pipeline.config import ANALYSIS, CARD_SYNTH_PCT, DATA_V10, DOMAINS

SAMPLE = 500


def load(d):
    return pd.concat([pd.read_csv(DATA_V10 / d.key / f"{s}.csv", dtype=str, keep_default_na=False)
                      for s in ("train", "test")], ignore_index=True)


def main():
    texts = {d.key: [norm(t) for t in load(d).text] for d in DOMAINS}
    rows = []
    for script in ("Nastaliq", "Roman"):
        doms = [d for d in DOMAINS if d.script == script]
        # cross-domain index over every row of the script
        lsh, hashes, grams, owner = MinHashLSH(threshold=JACCARD, num_perm=NUM_PERM), [], [], []
        for d in doms:
            for t in texts[d.key]:
                h, g = minhash(t)
                lsh.insert(str(len(owner)), h)
                hashes.append(h); grams.append(g); owner.append(d.key)
        for d in doms:
            T = texts[d.key]
            comp, _ = near_dup_clusters(T)
            sizes = pd.Series(comp).value_counts()
            template_rate = float(pd.Series(comp).map(sizes).gt(1).mean())
            idx = [i for i, o in enumerate(owner) if o == d.key]
            seed = 0
            for i in idx:
                for c in lsh.query(hashes[i]):
                    j = int(c)
                    if owner[j] != d.key and jaccard(grams[i], grams[j]) >= JACCARD:
                        seed += 1
                        break
            rng = np.random.default_rng(0)
            sample = [T[i] for i in rng.choice(len(T), min(SAMPLE, len(T)), replace=False)]
            bigrams = [tuple(w[k:k + 2]) for w in (s.split() for s in sample) for k in range(len(w) - 1)]
            lens = np.array([len(t.split()) for t in T])
            y = load(d).label.value_counts()
            rows.append({
                "domain": f"{d.script[:3]} {d.sid}", "name": d.name, "n": len(T),
                "card_synth_pct": CARD_SYNTH_PCT[d.key],
                "template_rate": round(template_rate, 3),
                "distinct2": round(len(set(bigrams)) / max(1, len(bigrams)), 3),
                "len_mean": round(lens.mean(), 1), "len_cv": round(lens.std() / max(1e-9, lens.mean()), 2),
                "seed_match": round(seed / len(T), 3),
                "round_size": len(T) % 50 == 0,
                "class_counts": "/".join(map(str, y.sort_index().tolist())),
            })
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "synthetic_audit.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
