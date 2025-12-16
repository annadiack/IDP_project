#!/usr/bin/env python3
"""
Trains an MLP baseline on a recommended dataset (Fashion-MNIST or CIFAR-10)
with modern training tricks we have seen in research papers / strong baselines:
- weight decay (L2 regularization)
- dropout
- label smoothing
- cosine LR schedule
- early stopping
- optional mixup

Run examples:
  python3 train_mlp.py --dataset fashionmnist --epochs 10
  python3 train_mlp.py --dataset cifar10 --epochs 30 --mixup 0.2
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
    """Seed Python/NumPy/PyTorch RNGs for more reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism flags (may reduce speed a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Use GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset / Transforms

def build_transforms(dataset: str) -> Tuple[T.Compose, T.Compose, int, int, int]:
    """
    Create transforms for train/test + return dataset meta:
      (train_transform, test_transform, num_classes, input_channels, image_size)
    """
    dataset = dataset.lower()

    if dataset == "fashionmnist":
        # Grayscale 28x28, 10 classes
        mean, std = (0.2860,), (0.3530,)
        train_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        return train_tf, test_tf, 10, 1, 28

    if dataset == "cifar10":
        # RGB 32x32, 10 classes
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        # Augmentation often used in papers / strong baselines on CIFAR-10
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
    """Download dataset and split training set into train/val."""
    train_tf, test_tf, num_classes, in_ch, img_size = build_transforms(dataset)

    if dataset.lower() == "fashionmnist":
        full_train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_tf)

    else:  # cifar10
        full_train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    # Split train into train/val reproducibly
    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=g)

    return train_set, val_set, test_set, num_classes, in_ch, img_size


# MixUp (optional)

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp: create convex combinations of pairs of samples.
    Many papers report it improves generalization, especially on CIFAR-like data.

    Returns:
      mixed_x, y_a, y_b, lam  where final loss = lam*CE(pred,y_a) + (1-lam)*CE(pred,y_b)
    """
    if alpha <= 0.0:
        return x, y, y, 1.0

    # Sample mixing coefficient lambda from Beta(alpha, alpha)
    lam = np.random.beta(alpha, alpha)

    # Shuffle indices to pair each sample with a random other sample in batch
    idx = torch.randperm(x.size(0), device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_loss(logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float, label_smoothing: float) -> torch.Tensor:
    """Compute MixUp loss as weighted sum of two cross-entropies."""
    return lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing) + (1.0 - lam) * F.cross_entropy(
        logits, y_b, label_smoothing=label_smoothing
    )


# -----------------------------
# Model: MLP
# -----------------------------

class MLP(nn.Module):
    """Simple fully-connected network baseline: Flatten -> FC -> FC -> FC."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.net(x)


def count_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Train/Eval

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Compute average loss and accuracy on loader."""
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)  # for eval we typically use plain CE
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_n += x.size(0)

    return {"loss": total_loss / max(1, total_n), "acc": total_correct / max(1, total_n)}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float,
    mixup_alpha: float
) -> Dict[str, float]:
    """One training epoch with optional MixUp + label smoothing."""
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Optionally apply MixUp augmentation inside the batch
        if mixup_alpha > 0.0:
            x_mix, y_a, y_b, lam = mixup_batch(x, y, alpha=mixup_alpha)
            logits = model(x_mix)
            loss = mixup_loss(logits, y_a, y_b, lam, label_smoothing=label_smoothing)

            # Accuracy for MixUp is less “well-defined”; we approximate using original labels y
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
    """Keep track of best validation checkpoint."""
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
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Paper-style improvements / ablations
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor (e.g. 0.1).")
    parser.add_argument("--mixup", type=float, default=0.0,
                        help="MixUp alpha (e.g. 0.2). 0 disables MixUp.")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Stop if val acc doesn't improve for N epochs.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Load data
    train_set, val_set, test_set, num_classes, in_ch, img_size = load_data(
        args.dataset, args.data_dir, args.val_fraction, args.seed
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Build model
    input_dim = in_ch * img_size * img_size
    model = MLP(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=num_classes, dropout=args.dropout).to(device)
    print(f"MLP params: {count_params(model):,}")

    # Optimizer (weight_decay = L2 regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine LR schedule: strong default in many training recipes
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = BestState()
    patience_left = args.early_stop_patience

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, device,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup
        )
        val_m = evaluate(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr={lr_now:.5f} | "
            f"train_acc={train_m['acc']:.4f} val_acc={val_m['acc']:.4f} | "
            f"train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f}"
        )

        # Early stopping & best checkpoint
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

    # Restore best model and test
    if best.state_dict_cpu is not None:
        model.load_state_dict(best.state_dict_cpu)

    test_m = evaluate(model, test_loader, device)

    print("\n=== FINAL (Best Val Checkpoint) ===")
    print(f"Best val acc: {best.best_val_acc:.4f} at epoch {best.best_epoch}")
    print(f"Test acc:     {test_m['acc']:.4f}")
    print(f"Train time:   {train_time:.2f} seconds")


if __name__ == "__main__":
    main()
