"""Stage 3 — few-shot instruction-tuned LLMs (in-context learning, no weight updates).

For every (model, exemplar seed, task, source) the prompt holds K_PER_CLASS
demonstrations drawn from the SOURCE train split; it is then applied to a fixed
stratified sample of every TARGET test split. Fixes relative to the legacy
notebooks:
  * the model's chat template is used (instruct models were prompted raw before)
  * no whole-prompt truncation: legacy code cut prompts at 1,024 tokens from the
    right, which deleted the test item on long inputs; now each text is capped at
    task.max_len tokens, the same budget the encoders get
  * raw generations are stored; outputs that name no label are scored as errors
    (label -1) and reported as the parse-failure rate, instead of being silently
    mapped to class 0
  * the evaluation sample of each target is fixed across sources, models and
    seeds, so all comparisons are paired

Usage:  python -m pipeline.run_llm [--models Qwen2.5-7B] [--tasks N-SA] [--seeds 13] [--n-eval 300]
"""
import argparse
import copy
import gc
import hashlib
import os
import re
import time

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pipeline.common import cell_path, load_split, scores, split_pair, to_list, write_cell
from pipeline.config import DOMAIN_BY_KEY, EXEMPLAR_SEEDS, LLM_MODELS, TASKS, TASK_BY_ID

K_PER_CLASS = 5
MAX_NEW_TOKENS = 6
BATCH = 8
BATCH_CACHED = 16

INSTRUCTIONS = {
    "Sentiment / polarity": "Classify the sentiment of the text.",
    "Abusive language": "Decide whether the text is abusive or hateful.",
    "Fake news": "Decide whether the news text is real or fake.",
    "QA pair validation": "Decide whether the answer is a valid answer to the question.",
}


def eval_sample(df, n):
    """Fixed stratified sample of a target test split (seed 0), shared by every run.
    Column `row` keeps each item's position in the full test split, so fine-tuned
    predictions can be scored on exactly the same items (paired comparisons)."""
    df = df.reset_index(drop=True).assign(row=lambda d: d.index)
    if len(df) <= n:
        return df
    part, _ = train_test_split(df, train_size=n, stratify=df.y, random_state=0)
    return part.sort_values("row").reset_index(drop=True)


def clip(tok, text, n):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= n else tok.decode(ids[:n]) + " …"


def render(tok, task, text):
    if task.pair_input:
        q, a = split_pair(text)
        half = task.max_len // 2
        return f"Question: {clip(tok, q, half)}\nAnswer: {clip(tok, a, half)}"
    return f"Text: {clip(tok, text, task.max_len)}"


def build_messages(tok, task, demos, item_rendered):
    """Chat messages for one test item; `item_rendered` is the output of render()."""
    names = task.label_names
    options = ", ".join(names[:-1]) + f" or {names[-1]}"
    header = (f"{INSTRUCTIONS[task.task_type]} The text is Urdu written in "
              f"{'Nastaliq (Perso-Arabic) script' if task.script == 'Nastaliq' else 'Roman (Latin) script'}. "
              f"Answer with exactly one word: {options}.")
    shots = "\n\n".join(f"{render(tok, task, t)}\nLabel: {names[y]}" for t, y in demos)
    user = f"{header}\n\nExamples:\n\n{shots}\n\nNow classify:\n\n{item_rendered}\nLabel:"
    return [{"role": "user", "content": user}]


def parse(output, names):
    """Earliest whole-word mention of a label name wins ('Invalid' never matches 'Valid')."""
    low = output.lower()
    hits = []
    for i, n in enumerate(names):
        m = re.search(rf"(?<![a-z]){re.escape(n.lower())}(?![a-z])", low)
        if m:
            hits.append((m.start(), i))
    return min(hits)[1] if hits else -1


def draw_demos(train, task, seed):
    demos = []
    for c in range(len(task.labels)):
        pool = train[train.y == c]
        demos += list(pool.sample(n=min(K_PER_CLASS, len(pool)), random_state=seed)[["text", "y"]]
                      .itertuples(index=False, name=None))
    return pd.DataFrame(demos).sample(frac=1, random_state=seed).itertuples(index=False, name=None)


@torch.no_grad()
def generate(model, tok, prompts):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            out[i] = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return out


ITEM = "␟ITEM␟"   # marker splitting the rendered chat prompt into shared prefix / item part


def split_template(tok, task, demos):
    """Render the chat prompt once with a marker in place of the test item and split it:
    every prompt of one (source, seed) = prefix + render(item) + tail."""
    msgs = build_messages(tok, task, demos, ITEM)
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prefix, tail = full.split(ITEM)
    return prefix, tail


@torch.no_grad()
def generate_cached(model, tok, prefix, items, tail):
    """Greedy generation with the shared prefix encoded once. Each batch is laid out as
    [prefix | pad ... | item tokens]; positions come from the attention mask, so the
    item tokens continue directly after the prefix exactly as in an unpadded prompt."""
    dev = model.device
    p_ids = tok(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    base = model(p_ids, use_cache=True).past_key_values
    suffixes = [tok(x + tail, add_special_tokens=False).input_ids for x in items]
    order = sorted(range(len(items)), key=lambda i: len(suffixes[i]))
    out = [None] * len(items)
    for b in range(0, len(order), BATCH_CACHED):
        idx = order[b:b + BATCH_CACHED]
        L = max(len(suffixes[i]) for i in idx)
        ids = torch.full((len(idx), p_ids.shape[1] + L), tok.pad_token_id, dtype=torch.long, device=dev)
        mask = torch.zeros_like(ids)
        ids[:, :p_ids.shape[1]] = p_ids
        mask[:, :p_ids.shape[1]] = 1
        for r, i in enumerate(idx):
            s = torch.tensor(suffixes[i], device=dev)
            ids[r, -len(s):] = s
            mask[r, -len(s):] = 1
        cache = copy.deepcopy(base)
        cache.batch_repeat_interleave(len(idx))
        gen = model.generate(input_ids=ids, attention_mask=mask, past_key_values=cache,
                             max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for r, i in enumerate(idx):
            out[i] = tok.decode(gen[r, ids.shape[1]:], skip_special_tokens=True).strip()
    return out


def load(model_name):
    hf_id = LLM_MODELS[model_name]
    tok = AutoTokenizer.from_pretrained(hf_id, token=os.environ.get("HF_TOKEN"))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(hf_id, quantization_config=q, device_map="cuda:0",
                                                 torch_dtype=torch.bfloat16,
                                                 token=os.environ.get("HF_TOKEN"))
    model.eval()
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(LLM_MODELS))
    ap.add_argument("--tasks", nargs="*", default=[t.tid for t in TASKS])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(EXEMPLAR_SEEDS))
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--limit-sources", type=int, default=None, help="debug: first k sources only")
    ap.add_argument("--check", type=int, default=0, help="debug: compare cached vs uncached on k items, then exit")
    a = ap.parse_args()

    for model_name in a.models:
        tok = model = None
        for seed in a.seeds:
            for tid in a.tasks:
                task = TASK_BY_ID[tid]
                srcs = task.domains[:a.limit_sources] if a.limit_sources else task.domains
                todo = [(s, t) for s in srcs for t in task.domains
                        if not cell_path("llm", task, model_name, seed, s, t).exists()]
                if not todo:
                    continue
                if model is None:
                    tok, model = load(model_name)
                tests = {k: eval_sample(load_split(k, "test", task), a.n_eval) for k in task.domains}
                for src in srcs:
                    tgts = [t for s, t in todo if s == src]
                    if not tgts:
                        continue
                    demos = list(draw_demos(load_split(src, "train", task), task, seed))
                    prefix, tail = split_template(tok, task, demos)
                    for tgt in tgts:
                        t0 = time.time()
                        te = tests[tgt]
                        items = [render(tok, task, x) for x in te.text]
                        raw = generate_cached(model, tok, prefix, items, tail)
                        if a.check:
                            full = [prefix + x + tail for x in items[:a.check]]
                            ref = generate(model, tok, full)
                            agree = sum(parse(x, task.label_names) == parse(y, task.label_names)
                                        for x, y in zip(raw, ref))
                            print(f"  CHECK cached vs uncached label agreement: {agree}/{len(ref)}", flush=True)
                            return
                        pred = [parse(r, task.label_names) for r in raw]
                        rec = {
                            "family": "llm", "task": tid, "model": model_name,
                            "hf_id": LLM_MODELS[model_name], "seed": seed,
                            "source": src, "target": tgt, "in_domain": src == tgt,
                            "k_per_class": K_PER_CLASS,
                            "demos_md5": hashlib.md5("\x1f".join(d[0] for d in demos).encode()).hexdigest(),
                            "parse_failures": sum(p == -1 for p in pred),
                            "prefix_tokens": len(tok(prefix, add_special_tokens=False).input_ids),
                            "seconds": round(time.time() - t0, 1),
                            **scores(te.y.values, pred, len(task.labels)),
                            "rows": to_list(te.row.values),
                            "y_true": to_list(te.y.values), "y_pred": pred, "raw": raw,
                        }
                        write_cell(cell_path("llm", task, model_name, seed, src, tgt), rec)
                        print(f"  {model_name} s{seed} {tid} {DOMAIN_BY_KEY[src].sid}->{DOMAIN_BY_KEY[tgt].sid} "
                              f"F1={rec['macro_f1']:.4f} parse_fail={rec['parse_failures']}/{rec['n']} "
                              f"{rec['seconds']}s", flush=True)
        if model is not None:
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
