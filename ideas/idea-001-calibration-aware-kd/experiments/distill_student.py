"""
distill_student.py – Train a student model via knowledge distillation.

Usage:
    # Standard KD
    python distill_student.py --dataset cifar10 \
        --teacher_arch resnet56 --student_arch resnet20 \
        --method standard_kd --temperature 4 --alpha 0.9 --seed 42

    # CA-KD (calibration-aware)
    python distill_student.py --dataset cifar10 \
        --teacher_arch resnet56 --student_arch resnet20 \
        --method cakd --temperature 4 --alpha 0.9 --lambda_cal 0.1 --seed 42

Outputs:
    results/student_<dataset>_<student>_<method>_seed<seed>.pth
    results/student_<dataset>_<student>_<method>_seed<seed>.json
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from losses import standard_kd_loss, cakd_loss
from models import get_model
from train_teacher import get_dataloaders, set_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_teacher(arch: str, dataset: str, num_classes: int, seed: int, device: str):
    ckpt_path = os.path.join(RESULTS_DIR, f"teacher_{dataset}_{arch}_seed{seed}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {ckpt_path}\n"
            f"Run train_teacher.py first."
        )
    model = get_model(arch, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def train_epoch_kd(student, teacher, loader, optimizer, loss_fn, device):
    student.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        optimizer.zero_grad()
        student_logits = student(inputs)
        loss = loss_fn(student_logits, teacher_logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += student_logits.argmax(1).eq(labels).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += inputs.size(0)
    return 100.0 * correct / total


def build_loss_fn(args):
    if args.method == "standard_kd":
        def loss_fn(s_logits, t_logits, labels):
            return standard_kd_loss(s_logits, t_logits, labels,
                                    temperature=args.temperature, alpha=args.alpha)
    elif args.method == "cakd":
        def loss_fn(s_logits, t_logits, labels):
            return cakd_loss(s_logits, t_logits, labels,
                             temperature=args.temperature, alpha=args.alpha,
                             lambda_cal=args.lambda_cal)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    return loss_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--teacher_arch", default="resnet56")
    parser.add_argument("--student_arch", default="resnet20")
    parser.add_argument("--method",       choices=["standard_kd", "cakd"], default="standard_kd")
    parser.add_argument("--temperature",  type=float, default=4.0)
    parser.add_argument("--alpha",        type=float, default=0.9)
    parser.add_argument("--lambda_cal",   type=float, default=0.1,
                        help="Calibration loss weight (CA-KD only)")
    parser.add_argument("--epochs",       type=int, default=200)
    parser.add_argument("--lr",           type=float, default=0.1)
    parser.add_argument("--batch",        type=int, default=128)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Method: {args.method}")

    train_loader, test_loader, num_classes = get_dataloaders(args.dataset, args.batch)

    teacher = load_teacher(args.teacher_arch, args.dataset, num_classes, args.seed, device)
    student = get_model(args.student_arch, num_classes=num_classes).to(device)

    loss_fn   = build_loss_fn(args)
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    method_tag = args.method if args.method == "standard_kd" else f"cakd_lam{args.lambda_cal}"
    stem = f"student_{args.dataset}_{args.student_arch}_{method_tag}_seed{args.seed}"

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch_kd(student, teacher, train_loader, optimizer,
                                               loss_fn, device)
        test_acc = evaluate(student, test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | "
              f"train {train_acc:.2f}% | test {test_acc:.2f}% | {elapsed:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), os.path.join(RESULTS_DIR, f"{stem}.pth"))

    metrics = {"dataset": args.dataset, "teacher_arch": args.teacher_arch,
               "student_arch": args.student_arch, "method": args.method,
               "temperature": args.temperature, "alpha": args.alpha,
               "lambda_cal": args.lambda_cal if args.method == "cakd" else None,
               "seed": args.seed, "best_test_acc": best_acc, "epochs": args.epochs}
    with open(os.path.join(RESULTS_DIR, f"{stem}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nBest test accuracy: {best_acc:.2f}%")
    print(f"Results saved to {RESULTS_DIR}/{stem}.json")


if __name__ == "__main__":
    main()
