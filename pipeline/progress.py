"""Print how many result cells exist versus the full matrix.  python -m pipeline.progress"""
from pipeline.config import EXEMPLAR_SEEDS, FT_MODELS, LLM_MODELS, RUNS, SEEDS, TASKS

for family, models, seeds in (("ft", FT_MODELS, SEEDS), ("llm", LLM_MODELS, EXEMPLAR_SEEDS)):
    for m in models:
        done = sum(len(list((RUNS / family / t.tid / m).glob("seed*/*.json"))) for t in TASKS)
        total = sum(len(t.domains) ** 2 for t in TASKS) * len(seeds)
        print(f"{family:3s} {m:14s} {done:5d}/{total}  ({100 * done / total:5.1f}%)")
