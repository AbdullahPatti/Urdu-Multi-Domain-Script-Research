# Domain Robustness of Multilingual NLP Models Across Urdu and Roman Urdu Scripts

Code, results and paper for a study of how Urdu text classifiers behave when the data source
changes while the language, script and task stay the same.

- **Paper:** `paper/main.pdf` (LaTeX source in `paper/`)
- **Benchmark (v1.1):** [Hugging Face](https://huggingface.co/datasets/abdullaharoon/Urdu-Multi-Domain-Benchmark)
  and [Zenodo](https://doi.org/10.5281/zenodo.23053495) (DOI 10.5281/zenodo.23053495)
- **Authors:** Muhammad Abdullah Haroon and Maryam Bashir, FAST School of Computing, FAST-NUCES Lahore

## The benchmark

26 labelled datasets (188,309 examples) in Nastaliq and Roman Urdu, grouped into 7 transfer
matrices over four task types: sentiment and polarity, abusive language, fake news, and
question-answer pair validation. All domains of a matrix share one label space, giving 80
source-to-target shifts. The domains were selected from 33 candidates by stated criteria (no
duplicated sources, no labels predictable from the origin of an item, at least 100 test items),
and every split is free of exact and near-duplicate train-test overlap.

## What is evaluated

Following Calderon et al. (2024): fine-tuned XLM-R and mBERT (3 seeds) and few-shot
Llama-3.1-8B, Qwen2.5-7B and Mistral-7B (3 demonstration draws), on every shift, with in-domain,
cross-domain and target in-domain scores, Source and Target Drop per shift, majority, random and
length-only baselines, and Holm-corrected paired bootstrap tests.

## Repository layout

| Path | Contents |
|---|---|
| `pipeline/config.py` | domains, transfer matrices, excluded candidates and their reasons |
| `pipeline/build_splits.py` | builds the benchmark splits (deterministic) |
| `pipeline/audit_synthetic.py`, `pipeline/audit_shortcuts.py` | generated-text and single-phrase shortcut audits |
| `pipeline/run_finetune.py`, `pipeline/run_llm.py` | fine-tuning and few-shot LLM runs |
| `pipeline/analyze.py`, `pipeline/errors.py` | aggregates, bootstrap tests, error analysis |
| `pipeline/make_tables.py`, `pipeline/figures.py` | every table, in-text number and figure of the paper |
| `pipeline/make_release.py`, `scripts/rebuild.py` | public release; rebuild of the four sources that state no licence |
| `runs/` | per-cell predictions of every model, seed and shift |
| `analysis/` | aggregated results and audits |
| `Working/` | notebooks from an earlier stage of the project (not used for the reported results) |

## Reproducing the results

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate elsewhere
pip install torch transformers bitsandbytes accelerate scikit-learn pandas matplotlib datasketch openpyxl
# put the benchmark files in data/v1.1/ (download from Hugging Face and run rebuild.py from the release)
python -m pipeline.run_finetune
python -m pipeline.run_llm            # Llama and Mistral are gated on Hugging Face
python -m pipeline.analyze && python -m pipeline.errors && python -m pipeline.audit_shortcuts
python -m pipeline.make_tables --results && python -m pipeline.figures
```

Hardware used: one NVIDIA RTX A4000 (16 GB); about 3 GPU-hours of fine-tuning and 4 GPU-hours of
LLM inference.

## Licences

Code: MIT. Each benchmark domain keeps the licence of its source; see `LICENSE.md` in the
Hugging Face / Zenodo release. Four public corpora whose authors state no licence are distributed
only through a rebuild script that downloads the authors' own files.

## Content warning

The abusive-language domains contain offensive, hateful and sectarian text.
