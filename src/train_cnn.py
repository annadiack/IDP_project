#!/usr/bin/env python3
"""
Trains a simple CNN on Fashion-MNIST or CIFAR-10 with:
- data augmentation (CIFAR-10)
- weight decay
- dropout
- label smoothing
- cosine LR schedule
- early stopping
- optional MixUp

Run examples:
  python3 train_cnn.py --dataset fashionmnist --epochs 10
  python3 train_cnn.py --dataset cifar10 --epochs 50 --mixup 0.2 --label-smoothing 0.1
"""

import argparse
import time
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T


# Reproducibility

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset / Transforms

def build_transforms(dataset: str) -> Tuple[T.Compose, T.Compose, int, int, int]:
    dataset = dataset.lower()

    if dataset == "fashionmnist":
        mean, std = (0.2860,), (0.3530,)
        train_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        return train_tf, test_tf, 10, 1, 28

    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        return train_tf, test_tf, 10, 3, 32

    raise ValueError("dataset must be fashionmnist or cifar10")


def load_data(
    dataset: str,
    data_dir: str,
    val_fraction: float,
    seed: int
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, int, int, int]:
    train_tf, test_tf, num_classes, in_ch, img_size = build_transforms(dataset)

    if dataset.lower() == "fashionmnist":
        full_train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_tf)
    else:
        full_train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=g)

    return train_set, val_set, test_set, num_classes, in_ch, img_size


# MixUp (optional)

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_loss(logits, y_a, y_b, lam, label_smoothing: float):
    return lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing) + (1 - lam) * F.cross_entropy(
        logits, y_b, label_smoothing=label_smoothing
    )


# Model: Simple CNN (with BatchNorm)

class SimpleCNN(nn.Module):
    """
    A compact CNN that works for both:
      - Fashion-MNIST (1x28x28)
      - CIFAR-10 (3x32x32)

    Includes BatchNorm (common in many CNN papers/training recipes).
    """
    def __init__(self, in_ch: int, num_classes: int, dropout: float):
        super().__init__()

        # Feature extractor block 1
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Feature extractor block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout(p=dropout)

        # We'll create the classifier lazily once we know flattened feature size
        self.fc1 = None
        self.fc2 = None
        self.num_classes = num_classes

    def _build_classifier_if_needed(self, feat_dim: int, device: torch.device):
        """Build final MLP head once we know the flattened dimension."""
        if self.fc1 is None:
            self.fc1 = nn.Linear(feat_dim, 128).to(device)
            self.fc2 = nn.Linear(128, self.num_classes).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x)), inplace=True))

        # Conv block 2: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x)), inplace=True))

        # Flatten for classifier head
        x = torch.flatten(x, start_dim=1)

        # Build classifier head dynamically (works for 7x7 or 8x8 spatial sizes)
        self._build_classifier_if_needed(x.shape[1], x.device)

        # MLP head with dropout regularization
        x = self.drop(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Train/Eval

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_n += x.size(0)

    return {"loss": total_loss / max(1, total_n), "acc": total_correct / max(1, total_n)}


def train_one_epoch(model, loader, optimizer, device, label_smoothing: float, mixup_alpha: float) -> Dict[str, float]:
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if mixup_alpha > 0.0:
            x_mix, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
            logits = model(x_mix)
            loss = mixup_loss(logits, y_a, y_b, lam, label_smoothing)
            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += correct
        total_n += x.size(0)

    return {"loss": total_loss / max(1, total_n), "acc": total_correct / max(1, total_n)}


@dataclass
class BestState:
    best_val_acc: float = -1.0
    best_epoch: int = -1
    state_dict_cpu: Optional[Dict[str, torch.Tensor]] = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fashionmnist", "cifar10"], default="fashionmnist")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # “paper-style” knobs
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_set, val_set, test_set, num_classes, in_ch, _ = load_data(
        args.dataset, args.data_dir, args.val_fraction, args.seed
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SimpleCNN(in_ch=in_ch, num_classes=num_classes, dropout=args.dropout).to(device)
    print(f"CNN params (initial): {count_params(model):,}  (may increase after first forward due to lazy head)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = BestState()
    patience_left = args.early_stop_patience
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, device, args.label_smoothing, args.mixup)
        val_m = evaluate(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        # After the first epoch, the lazy classifier is built, so param count becomes stable
        if epoch == 1:
            print(f"CNN params (final): {count_params(model):,}")

        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr={lr_now:.5f} | "
            f"train_acc={train_m['acc']:.4f} val_acc={val_m['acc']:.4f} | "
            f"train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f}"
        )

        if val_m["acc"] > best.best_val_acc:
            best.best_val_acc = val_m["acc"]
            best.best_epoch = epoch
            best.state_dict_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (no val acc improvement).")
                break

    train_time = time.time() - t0

    if best.state_dict_cpu is not None:
        model.load_state_dict(best.state_dict_cpu)

    test_m = evaluate(model, test_loader, device)

    print("\n=== FINAL (Best Val Checkpoint) ===")
    print(f"Best val acc: {best.best_val_acc:.4f} at epoch {best.best_epoch}")
    print(f"Test acc:     {test_m['acc']:.4f}")
    print(f"Train time:   {train_time:.2f} seconds")


if __name__ == "__main__":
    main()
