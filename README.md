# Domain Robustness of Multilingual NLP Models Across Urdu and Roman Urdu Scripts

[![Dataset on Hugging Face](https://img.shields.io/badge/dataset-Hugging%20Face-1f4e8c)](https://huggingface.co/datasets/abdullaharoon/Urdu-Multi-Domain-Benchmark)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23053495-1f4e8c)](https://doi.org/10.5281/zenodo.23053495)
[![Paper](https://img.shields.io/badge/paper-PDF-1f4e8c)](paper/main.pdf)

**Muhammad Abdullah Haroon and Maryam Bashir**
FAST School of Computing, National University of Computer and Emerging Sciences (FAST-NUCES), Lahore

Most Urdu text classifiers are trained and tested on one corpus, so it is rarely known how they
behave when the data source changes. This repository contains the benchmark construction, the
experiments and the paper of a study that measures this directly: models are trained (or
prompted) on one domain of a task and tested on every other domain of the same task, in both
Nastaliq and Roman Urdu.

## Main findings

- **Fine-tuned encoders lose 13.5 to 47.6 macro-F1 points** when moved to another domain of the
  same task. In 83 to 100% of shifts a model trained on the target domain does better there, so
  the loss is a robustness failure rather than a move to a harder domain.
- **Few-shot LLMs lose far less.** Their cross-domain F1 is significantly higher than that of a
  fine-tuned encoder in 30 of 42 paired comparisons (Holm-corrected bootstrap), significantly
  lower in 4, all of them in QA validation, and not significantly different in 8.
- **The largest failures come from label definitions, not vocabulary.** In the worst shifts the
  source model often assigns one class to almost every item, because two domains attach the
  same label to different kinds of text (for example, mild criticism labelled as hate in one
  domain and only insults in another).
- **Fake news does not transfer** for either family: fine-tuned models fall below chance across
  domains, and the LLMs stay close to chance and to a classifier that uses only text length.
- The conclusions hold when LLM-generated domains are excluded.

## The benchmark

26 datasets, 188,309 labelled examples, 7 transfer matrices, 80 source-to-target shifts. All
domains of a matrix share one label space.

| Script | Transfer matrix | Domains | Labels |
|---|---|---:|---|
| Nastaliq | Sentiment | 3 | negative, positive |
| Nastaliq | Hate speech | 4 | normal, hate/offensive |
| Nastaliq | Fake news | 2 | fake, real |
| Nastaliq | QA pair validation | 4 | invalid, valid |
| Roman Urdu | Sentiment | 3 | negative, positive |
| Roman Urdu | Cyber abuse | 4 | normal, abusive |
| Roman Urdu | 3-class polarity | 6 | negative, neutral, positive |

The domains were selected from 33 candidates by three criteria: no domain may duplicate
another, no label may be predictable from the origin of an item, and every domain needs at
least 100 test items. Every split is free of exact and near-duplicate train-test overlap
(cluster-aware splits for LLM-generated domains, question-grouped splits for QA).

**Load a domain**

```python
from datasets import load_dataset
ds = load_dataset("abdullaharoon/Urdu-Multi-Domain-Benchmark", "R-SA_SA-B")   # Roman sentiment, domain B
```

Four domains come from public corpora whose authors state no licence. Their text is not
redistributed; `rebuild.py` in the dataset release downloads the authors' own files and
reconstructs our splits.

## Results at a glance

Macro-F1, ranges over models of each family (means over 3 seeds or demonstration draws).
SS = in-domain, ST = cross-domain.

| Matrix | Fine-tuned SS | Fine-tuned ST | Few-shot LLM SS | Few-shot LLM ST |
|---|---|---|---|---|
| Nastaliq sentiment | 0.80-0.81 | 0.57-0.64 | 0.69-0.79 | 0.68-0.79 |
| Nastaliq hate speech | 0.90 | 0.52-0.57 | 0.65-0.75 | 0.54-0.58 |
| Nastaliq fake news | 0.78-0.86 | 0.36-0.38 | 0.51-0.61 | 0.54-0.58 |
| Nastaliq QA validation | 0.88 | 0.63-0.72 | 0.62-0.82 | 0.58-0.69 |
| Roman sentiment | 0.85 | 0.68-0.72 | 0.80-0.82 | 0.78-0.80 |
| Roman cyber abuse | 0.99-1.00 | 0.80-0.84 | 0.87-0.88 | 0.83-0.87 |
| Roman 3-class polarity | 0.74-0.76 | 0.38-0.45 | 0.72-0.75 | 0.65-0.71 |

Fine-tuned: XLM-R, mBERT. Few-shot: Llama-3.1-8B, Qwen2.5-7B, Mistral-7B (4-bit). Full tables,
Target Drop analysis, confidence intervals and error analysis are in the paper.

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | LaTeX source, figures, generated tables and the compiled `main.pdf` |
| `pipeline/config.py` | domains, transfer matrices, excluded candidates and the reason for each |
| `pipeline/build_splits.py` | builds the benchmark splits (deterministic) |
| `pipeline/audit_synthetic.py`, `pipeline/audit_shortcuts.py` | generated-text and single-phrase shortcut audits |
| `pipeline/run_finetune.py`, `pipeline/run_llm.py` | fine-tuning and few-shot LLM experiments |
| `pipeline/analyze.py`, `pipeline/errors.py` | aggregates, bootstrap tests, error analysis |
| `pipeline/make_tables.py`, `pipeline/figures.py` | every table, in-text number and figure of the paper |
| `pipeline/make_release.py`, `scripts/rebuild.py` | dataset release and rebuild of the four unlicensed sources |
| `scripts/make_arxiv.py` | builds the arXiv source package |
| `runs/` | predictions of every model, seed and shift (1,590 result cells) |
| `analysis/` | aggregated results and audit tables |

## Reproducing the results

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch transformers bitsandbytes accelerate scikit-learn pandas matplotlib datasketch openpyxl huggingface_hub

# 1. Get the data: download the release from Hugging Face or Zenodo into data/v1.1/
#    and run its rebuild.py for the four domains distributed as manifests.
# 2. Run the experiments (Llama and Mistral are gated models on Hugging Face).
python -m pipeline.run_finetune
python -m pipeline.run_llm
# 3. Regenerate every number, table and figure of the paper.
python -m pipeline.analyze
python -m pipeline.errors
python -m pipeline.audit_shortcuts
python -m pipeline.make_tables --results
python -m pipeline.figures
```

Every number in the paper is produced by these scripts; none is typed by hand. The reported run
used one NVIDIA RTX A4000 (16 GB): about 3 GPU-hours of fine-tuning and 4 GPU-hours of LLM
inference. Results are resumable: finished cells in `runs/` are skipped.

## Citation

```bibtex
@misc{haroon2026urdurobustness,
  title  = {Domain Robustness of Multilingual {NLP} Models Across {U}rdu and {R}oman {U}rdu Scripts},
  author = {Haroon, Muhammad Abdullah and Bashir, Maryam},
  year   = {2026},
  note   = {Dataset: \url{https://doi.org/10.5281/zenodo.23053495}}
}
```

Please also cite the original corpora listed in `domains.csv` of the dataset release.

## Licence

Code in this repository: MIT (see `LICENSE`). Each benchmark domain keeps the licence of its source (see
`LICENSE.md` in the dataset release); our splits, manifests and audits are released under
CC BY-NC 4.0. LLM-generated domains were produced with xAI Grok and are attributed to it.

## Content warning

The abusive-language and hate-speech domains contain offensive, hateful and sectarian text.
They are released for research on moderation; classifiers trained on them should not be
deployed without human review.
