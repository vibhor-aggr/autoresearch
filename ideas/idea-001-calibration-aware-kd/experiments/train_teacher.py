"""
train_teacher.py – Train a ResNet teacher on CIFAR-10 or CIFAR-100.

Usage:
    python train_teacher.py --dataset cifar10 --arch resnet56 --epochs 200 --seed 42

Outputs:
    results/teacher_<dataset>_<arch>_seed<seed>.pth   model checkpoint
    results/teacher_<dataset>_<arch>_seed<seed>.json  final metrics
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
from torchvision import datasets, transforms
from torchvision.models import resnet50

from models import get_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(dataset: str, batch_size: int = 128):
    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        DataClass = datasets.CIFAR10
    elif dataset == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        DataClass = datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_root = os.path.join(os.path.dirname(__file__), "data")
    train_ds = DataClass(data_root, train=True,  transform=train_transform, download=True)
    test_ds  = DataClass(data_root, train=False, transform=test_transform,  download=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=256,       shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, num_classes


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--arch",    default="resnet56")
    parser.add_argument("--epochs",  type=int, default=200)
    parser.add_argument("--lr",      type=float, default=0.1)
    parser.add_argument("--batch",   type=int, default=128)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, test_loader, num_classes = get_dataloaders(args.dataset, args.batch)
    model = get_model(args.arch, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_acc = 0.0
    stem = f"teacher_{args.dataset}_{args.arch}_seed{args.seed}"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | "
              f"train {train_acc:.2f}% | test {test_acc:.2f}% | {elapsed:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{stem}.pth"))

    metrics = {"dataset": args.dataset, "arch": args.arch, "seed": args.seed,
               "best_test_acc": best_acc, "epochs": args.epochs}
    with open(os.path.join(RESULTS_DIR, f"{stem}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nBest test accuracy: {best_acc:.2f}%")
    print(f"Results saved to {RESULTS_DIR}/{stem}.json")


if __name__ == "__main__":
    main()
