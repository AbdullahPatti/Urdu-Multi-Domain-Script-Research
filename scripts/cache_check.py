"""Check that prefix-KV caching does not change LLM predictions: for every model,
generate labels for the first N items of one fake-news shift with and without the
cached prefix and count agreements. Result is appended to logs/cache_check.log."""
import sys, datetime
from pipeline.common import load_split
from pipeline.config import LLM_MODELS, TASK_BY_ID
from pipeline.config import load_env
from pipeline import run_llm as R

N = 32
load_env()
task = TASK_BY_ID["N-FN"]
src, tgt = task.domains[0], task.domains[1]
out = []
for name in LLM_MODELS:
    tok, model = R.load(name)
    demos = list(R.draw_demos(load_split(src, "train", task), task, 13))
    prefix, tail = R.split_template(tok, task, demos)
    te = R.eval_sample(load_split(tgt, "test", task), 300)
    items = [R.render(tok, task, x) for x in te.text[:N]]
    cached = R.generate_cached(model, tok, prefix, items, tail)
    full = R.generate(model, tok, [prefix + x + tail for x in items])
    agree = sum(R.parse(a, task.label_names) == R.parse(b, task.label_names) for a, b in zip(cached, full))
    out.append(f"{name}: {agree}/{N}")
    print(out[-1], flush=True)
    del model, tok
    import gc, torch; gc.collect(); torch.cuda.empty_cache()
with open("logs/cache_check.log", "a", encoding="utf-8") as f:
    f.write(f"[{datetime.datetime.now():%Y-%m-%dT%H:%M}] N-FN FN-A->FN-B seed 13: " + "; ".join(out) + "\n")
