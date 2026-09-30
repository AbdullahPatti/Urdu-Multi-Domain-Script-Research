"""Stage 2 — fine-tuned encoders (XLM-R base, mBERT cased).

One model is trained per (task, model, seed, source domain) and evaluated on the
test split of every domain in the task: the diagonal gives SS/TT, the off-diagonal
cells give ST. Training depends only on the source, so this is n trainings per
task instead of n^2 (the legacy notebooks retrained for every pair).

Usage:  python -m pipeline.run_finetune [--tasks N-SA R-CA] [--models XLM-R] [--seeds 13]
Finished cells are skipped, so the command can be re-run after an interruption.
"""
import argparse
import copy
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

from pipeline.common import cell_path, load_split, scores, split_pair, to_list, write_cell
from pipeline.config import DOMAIN_BY_KEY, FT_MODELS, SEEDS, TASKS, TASK_BY_ID

MAX_TRAIN = 8000   # stratified cap on source training rows (fixed sample, seed 0)
EPOCHS = 5
PATIENCE = 2
LR = 2e-5
BATCH = 16
VAL_FRAC = 0.1


class Encoded(Dataset):
    def __init__(self, df, tok, task):
        if task.pair_input:
            q, a = zip(*df.text.map(split_pair))
            self.enc = tok(list(q), list(a), truncation="only_second", max_length=task.max_len)
        else:
            self.enc = tok(df.text.tolist(), truncation=True, max_length=task.max_len)
        self.y = df.y.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


class KeepBest(TrainerCallback):
    """Early stopping on validation macro-F1, holding the best weights in CPU memory
    (equivalent to load_best_model_at_end without writing ~1 GB checkpoints per epoch)."""
    def __init__(self, patience):
        self.best, self.state, self.epoch, self.bad, self.patience = -1.0, None, 0, 0, patience

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        if metrics["eval_macro_f1"] > self.best:
            self.best, self.bad, self.epoch = metrics["eval_macro_f1"], 0, state.epoch
            self.state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            self.bad += 1
            if self.bad >= self.patience:
                control.should_training_stop = True


def cap(df, n=MAX_TRAIN):
    if len(df) <= n:
        return df
    part, _ = train_test_split(df, train_size=n, stratify=df.y, random_state=0)
    return part


def metric_fn(eval_pred):
    logits, labels = eval_pred
    return {"macro_f1": f1_score(labels, logits.argmax(-1), average="macro")}


def run_source(task, model_name, seed, src, tests):
    targets = [t for t in task.domains
               if not cell_path("ft", task, model_name, seed, src, t).exists()]
    if not targets:
        print(f"  [skip] {task.tid} {model_name} seed{seed} {DOMAIN_BY_KEY[src].sid}")
        return
    set_seed(seed)
    t0 = time.time()
    train_full = cap(load_split(src, "train", task))
    tr, va = train_test_split(train_full, test_size=VAL_FRAC, stratify=train_full.y, random_state=seed)

    hf_id = FT_MODELS[model_name]
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=len(task.labels))
    keep = KeepBest(PATIENCE)
    args = TrainingArguments(
        output_dir=str(cell_path("ft", task, model_name, seed, src, src).parent / "_tmp"),
        num_train_epochs=EPOCHS, learning_rate=LR, warmup_ratio=0.1, weight_decay=0.01,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=64,
        eval_strategy="epoch", save_strategy="no", logging_strategy="no",
        bf16=torch.cuda.is_available(), report_to="none", seed=seed, data_seed=seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=Encoded(tr, tok, task),
                      eval_dataset=Encoded(va, tok, task), compute_metrics=metric_fn,
                      data_collator=DataCollatorWithPadding(tok), callbacks=[keep])
    trainer.train()
    model.load_state_dict(keep.state)
    train_secs = time.time() - t0

    for tgt in targets:
        te = tests[tgt]
        logits = trainer.predict(Encoded(te, tok, task)).predictions
        pred = logits.argmax(-1)
        rec = {
            "family": "ft", "task": task.tid, "model": model_name, "hf_id": hf_id, "seed": seed,
            "source": src, "target": tgt, "in_domain": src == tgt,
            "train_rows": len(tr), "val_rows": len(va), "best_val_macro_f1": keep.best,
            "best_epoch": keep.epoch, "train_seconds": round(train_secs, 1),
            **scores(te.y.values, pred, len(task.labels)),
            "y_true": to_list(te.y.values), "y_pred": to_list(pred),
        }
        write_cell(cell_path("ft", task, model_name, seed, src, tgt), rec)
        print(f"  {task.tid} {model_name} s{seed} {DOMAIN_BY_KEY[src].sid}->{DOMAIN_BY_KEY[tgt].sid}"
              f"  F1={rec['macro_f1']:.4f}  (n={rec['n']})")
    del trainer, model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=[t.tid for t in TASKS])
    ap.add_argument("--models", nargs="*", default=list(FT_MODELS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    a = ap.parse_args()
    for seed in a.seeds:                       # seed-major: one full seed finishes first
        for tid in a.tasks:
            task = TASK_BY_ID[tid]
            tests = {k: load_split(k, "test", task) for k in task.domains}
            for model_name in a.models:
                for src in task.domains:
                    run_source(task, model_name, seed, src, tests)


if __name__ == "__main__":
    main()
