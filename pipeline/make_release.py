"""Stage 7 — assemble the public release (Hugging Face + Zenodo) in release/v1.1/.

  data/<Script>/<Collection>/<Domain>/{train,test}.csv   redistributable domains
  manifests/<same path>/{train,test}.csv                 domains whose source states no
        licence: row order, SHA-1 of the normalised text, and our label; no text
  rebuild.py                                             downloads those sources from the
        authors' GitHub repositories and writes their train/test CSVs into data/
  release_sources.py                                     readers used by rebuild.py
  domains.csv                                            one row per domain: matrix, origin,
        source, URL, licence, sizes, distribution mode
  audits/                                                data, synthetic-text and shortcut audits
  README.md                                              dataset card (Hugging Face YAML header)

Usage:  python -m pipeline.make_release
"""
import hashlib
import shutil

import pandas as pd

from pipeline.config import ANALYSIS, DATA_V11, DOMAIN_BY_KEY, DROPPED_V11, ROOT, TASKS
from pipeline.release_sources import DOMAIN_SOURCE, REPOS, norm

OUT = ROOT / "release" / "v1.1"
DOI = "10.5281/zenodo.23053495"
REPO = "https://github.com/AbdullahPatti/Urdu-Multi-Domain-Script-Research"


def sha1(text):
    return hashlib.sha1(norm(text).encode("utf-8")).hexdigest()


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data").mkdir(parents=True)
    prov = pd.read_csv(ANALYSIS / "provenance_worksheet.csv", encoding="utf-8-sig")
    audit = pd.read_csv(ANALYSIS / "data_audit_v11.csv", encoding="utf-8-sig").set_index("key")
    rows = []
    for t in TASKS:
        for k in t.domains:
            d = DOMAIN_BY_KEY[k]
            p = prov[(prov.script == d.script) & (prov.domain_id == d.sid)].iloc[0]
            mode = "rebuild" if k in DOMAIN_SOURCE else "file"
            for split in ("train", "test"):
                df = pd.read_csv(DATA_V11 / k / f"{split}.csv", encoding="utf-8-sig", dtype=str,
                                 keep_default_na=False)
                if mode == "file":
                    dst = OUT / "data" / k / f"{split}.csv"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    df[["text", "label"]].to_csv(dst, index=False, encoding="utf-8")
                else:
                    dst = OUT / "manifests" / k / f"{split}.csv"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({"row": range(len(df)), "sha1": df.text.map(sha1), "label": df.label}) \
                        .to_csv(dst, index=False, encoding="utf-8")
            rows.append({
                "path": k, "script": d.script, "matrix": t.tid, "matrix_name": t.title, "id": d.sid,
                "domain": d.name, "origin": d.origin, "source": p.original_source,
                "url": p.url if isinstance(p.url, str) else "", "licence": p.licence,
                "n_train": int(audit.loc[k, "v1.1_train"]), "n_test": int(audit.loc[k, "v1.1_test"]),
                "labels": "|".join(t.labels), "distribution": mode,
                "source_repo": REPOS[DOMAIN_SOURCE[k]] if mode == "rebuild" else "",
            })
    dom = pd.DataFrame(rows)
    dom.to_csv(OUT / "domains.csv", index=False, encoding="utf-8")
    pd.DataFrame([{"path": k, "reason": why} for k, why in DROPPED_V11.items()]) \
        .to_csv(OUT / "excluded_candidates.csv", index=False, encoding="utf-8")

    (OUT / "audits").mkdir()
    for f in ("data_audit_v11.csv", "synthetic_audit.csv", "shortcut_audit.csv"):
        shutil.copy(ANALYSIS / f, OUT / "audits" / f)
    shutil.copy(ROOT / "pipeline" / "release_sources.py", OUT / "release_sources.py")
    shutil.copy(ROOT / "scripts" / "rebuild.py", OUT / "rebuild.py")
    (OUT / "README.md").write_text(card(dom), encoding="utf-8")
    lic = "\n".join(f"| {r.script} {r.id} | {r.licence} | {r.url or '-'} |" for r in dom.itertuples())
    (OUT / "LICENSE.md").write_text(
        "# Licences\n\nThis release combines corpora under different terms. Each domain keeps the licence "
        "of its source:\n\n| Domain | Licence | Source |\n|---|---|---|\n" + lic +
        "\n\n*none stated*: distributed only as a manifest plus `rebuild.py`, which downloads the "
        "authors' own files.\n*xAI terms*: generated with xAI Grok; attribute the output to Grok and do "
        "not use it to develop models that compete with xAI.\n*unknown*: public upload whose author "
        "could not be identified; released for research use and removed on request from a rights holder.\n\n"
        "Our own contributions (splits, manifests, audits, curation of generated data) are released "
        "under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/).\n", encoding="utf-8")
    print(dom.groupby("distribution").size().to_dict(), "domains;",
          dom.n_train.sum() + dom.n_test.sum(), "examples")


def card(dom):
    files = dom[dom.distribution == "file"]
    configs = "\n".join(
        f"- config_name: {r.matrix}_{r.id}\n  data_files:\n"
        f"  - split: train\n    path: \"data/{r.path}/train.csv\"\n"
        f"  - split: test\n    path: \"data/{r.path}/test.csv\""
        for r in files.itertuples())
    table = "\n".join(
        f"| {r.matrix} | {r.id} | {r.domain} | {r.origin} | {r.n_train:,} | {r.n_test:,} | {r.licence} | {r.distribution} |"
        for r in dom.itertuples())
    excluded = "\n".join(f"- `{k}`: {why}" for k, why in DROPPED_V11.items())
    return f"""---
license: other
license_name: per-domain
license_link: LICENSE.md
language:
- ur
task_categories:
- text-classification
tags:
- urdu
- roman-urdu
- domain-robustness
- benchmark
pretty_name: Urdu Multi-Domain Benchmark (v1.1)
size_categories:
- 100K<n<1M
configs:
{configs}
---

# Urdu Multi-Domain Benchmark (v1.1)

A benchmark for **domain robustness** of Urdu text classifiers: {len(dom)} datasets in two scripts
(Nastaliq and Roman Urdu), grouped into 7 transfer matrices over four task types (sentiment and
polarity, abusive language, fake news, question-answer pair validation). All domains of a matrix
share one label space, so every source-to-target pair is a valid domain shift.

- Paper: *Domain Robustness of Multilingual NLP Models Across Urdu and Roman Urdu Scripts*
  (Haroon and Bashir, FAST-NUCES).
- Code, audits and model predictions: {REPO}
- DOI: https://doi.org/{DOI}

> **Content warning.** The abusive-language and hate-speech domains contain offensive, hateful and
> sectarian text. Classifiers trained on these data should not be deployed without human review.

## What changed from v1.0

v1.1 supersedes v1.0 (DOI 10.5281/zenodo.22195610). Domains were selected by explicit inclusion
criteria, and every split is free of exact and near-duplicate train-test overlap:

- **Excluded candidates** (duplicated sources, labels predictable from the origin of an item,
  fewer than 100 test items):
{excluded}
- **Near-duplicate-aware splits.** LLM-generated domains were re-split so that clusters of
  template near-duplicates (character 5-gram Jaccard >= 0.8) do not straddle train and test.
- **Question-grouped QA splits.** No question (or near-duplicate question) occurs in both splits.
- **Cross-domain overlap** removed inside each matrix.

## Domains

| Matrix | ID | Domain | Origin | Train | Test | Licence | Distribution |
|---|---|---|---|---:|---:|---|---|
{table}

Origin: *organic* (public corpus of native text), *translated* (machine-translated from English),
*generated* (produced by the first author with xAI Grok and checked by hand), *uncertain*.
Full provenance and URLs: `domains.csv`.

## Getting the data

Domains marked **file** are included directly (`data/`), each as a config of this dataset:

```python
from datasets import load_dataset
ds = load_dataset("abdullaharoon/Urdu-Multi-Domain-Benchmark", "R-SA_SA-B")
```

Domains marked **rebuild** come from public corpora whose authors state no licence, so their text is
not redistributed here. `manifests/` holds the row order, a SHA-1 of each normalised text and our
label; `rebuild.py` downloads the authors' GitHub repositories and writes the train/test files:

```bash
pip install pandas openpyxl
python rebuild.py            # writes data/<domain>/{{train,test}}.csv for the four rebuild domains
```

The rebuild recovers every item except 2 of 21,285 ISE-Hate tweets, which differ in the authors'
current file; `rebuild.py` reports them.

## Licences

Each domain keeps the licence of its source (table above). Traced sources with a permissive
licence are released under MIT, CC BY 4.0 or ODbL 1.0. Four published corpora (Urdu Sentiment
Corpus, ISE-Hate, Bend the Truth, Ax-to-Grind Urdu) state no licence and are distributed through
`rebuild.py` only; please cite their authors. Generated domains were produced with xAI Grok; its
terms ask that such output be attributed to Grok and prohibit using it to develop models that
compete with xAI. {(dom.licence == "unknown").sum()} organic domains were obtained from public
uploads whose authors we could not identify; they are released
for research use and will be removed on request from a rights holder. Our own contributions
(splits, manifests, audits, generated-data curation) are released under CC BY-NC 4.0.

## Citation

Please cite the paper and the original corpora listed in `domains.csv`.
"""


if __name__ == "__main__":
    main()
