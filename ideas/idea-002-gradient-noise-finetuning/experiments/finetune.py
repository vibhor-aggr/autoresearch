"""
finetune.py – Fine-tune DistilBERT on a text classification task.

Usage:
    # Standard fine-tuning
    python finetune.py --dataset sst2 --n_per_class 50 --method standard --seed 42

    # Gradient-noise fine-tuning
    python finetune.py --dataset sst2 --n_per_class 50 --method gn --eta 1e-3 --gamma 0.55 --seed 42

    # Adaptive gradient-noise fine-tuning
    python finetune.py --dataset sst2 --n_per_class 50 --method gn_adaptive --eta 1e-3 --gamma 0.55 --seed 42

Outputs:
    results/run_<dataset>_<n_per_class>_<method>_seed<seed>.json
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from gradient_noise_optimizer import GradientNoiseOptimizer, AdaptiveGradientNoiseOptimizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODEL_NAME  = "distilbert-base-uncased"

DATASET_CONFIGS = {
    "sst2":    {"hf_name": "glue",    "config": "sst2",    "text_col": "sentence",  "label_col": "label"},
    "ag_news": {"hf_name": "ag_news", "config": None,       "text_col": "text",      "label_col": "label"},
    "trec":    {"hf_name": "trec",    "config": None,       "text_col": "text",      "label_col": "coarse_label"},
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subsample(dataset, n_per_class: int, label_col: str, seed: int):
    """Stratified subsample of n_per_class examples per class."""
    labels = [ex[label_col] for ex in dataset]
    num_classes = len(set(labels))
    n_total = n_per_class * num_classes

    sss = StratifiedShuffleSplit(n_splits=1, test_size=None,
                                 train_size=min(n_total, len(dataset)),
                                 random_state=seed)
    indices, _ = next(sss.split(range(len(dataset)), labels))
    return Subset(dataset, indices.tolist())


def collate_fn(batch, tokenizer, text_col, label_col, max_length=128):
    texts  = [ex[text_col]  for ex in batch]
    labels = [ex[label_col] for ex in batch]
    enc = tokenizer(texts, truncation=True, padding=True,
                    max_length=max_length, return_tensors="pt")
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
        preds          = outputs.logits.argmax(dim=-1)
        correct       += preds.eq(labels).sum().item()
        total         += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def build_optimizer(model, args):
    from torch.optim import AdamW
    base_opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.method == "standard":
        return base_opt
    elif args.method == "gn":
        return GradientNoiseOptimizer(base_opt, eta=args.eta, gamma=args.gamma)
    elif args.method == "gn_adaptive":
        return AdaptiveGradientNoiseOptimizer(base_opt, eta=args.eta, gamma=args.gamma)
    else:
        raise ValueError(f"Unknown method: {args.method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     choices=list(DATASET_CONFIGS), default="sst2")
    parser.add_argument("--n_per_class", type=int, default=50)
    parser.add_argument("--method",      choices=["standard", "gn", "gn_adaptive"], default="standard")
    parser.add_argument("--eta",         type=float, default=1e-3)
    parser.add_argument("--gamma",       type=float, default=0.55)
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--weight_decay",type=float, default=0.01)
    parser.add_argument("--batch",       type=int, default=16)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Dataset: {args.dataset} | Method: {args.method}")

    cfg = DATASET_CONFIGS[args.dataset]
    hf_ds = (load_dataset(cfg["hf_name"], cfg["config"])
             if cfg["config"] else load_dataset(cfg["hf_name"]))

    train_split = hf_ds["train"]
    test_split  = hf_ds["validation"] if "validation" in hf_ds else hf_ds["test"]

    train_sub = subsample(train_split, args.n_per_class, cfg["label_col"], args.seed)
    num_classes = len(set(ex[cfg["label_col"]] for ex in train_split))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes).to(device)

    def _collate(batch):
        return collate_fn(batch, tokenizer, cfg["text_col"], cfg["label_col"])

    train_loader = DataLoader(train_sub,   batch_size=args.batch, shuffle=True,  collate_fn=_collate)
    test_loader  = DataLoader(test_split,  batch_size=64,         shuffle=False, collate_fn=_collate)

    optimizer = build_optimizer(model, args)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer if args.method == "standard" else optimizer._base_optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    method_tag = args.method if args.method == "standard" else f"{args.method}_eta{args.eta}_g{args.gamma}"
    stem = f"run_{args.dataset}_n{args.n_per_class}_{method_tag}_seed{args.seed}"

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += outputs.loss.item() * labels.size(0)
            correct    += outputs.logits.argmax(-1).eq(labels).sum().item()
            total      += labels.size(0)

        train_acc = 100.0 * correct / total
        test_acc  = evaluate(model, test_loader, device)
        gen_gap   = train_acc - test_acc
        history.append({"epoch": epoch, "train_acc": train_acc,
                         "test_acc": test_acc, "gen_gap": gen_gap})
        print(f"Epoch {epoch:2d} | train {train_acc:.1f}% | test {test_acc:.1f}% | gap {gen_gap:.1f}%")

    final = history[-1]
    result = {
        "dataset": args.dataset, "n_per_class": args.n_per_class,
        "method": args.method, "eta": args.eta, "gamma": args.gamma,
        "seed": args.seed, "epochs": args.epochs,
        "final_train_acc": final["train_acc"],
        "final_test_acc":  final["test_acc"],
        "final_gen_gap":   final["gen_gap"],
        "history": history,
    }
    out_path = os.path.join(RESULTS_DIR, f"{stem}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
