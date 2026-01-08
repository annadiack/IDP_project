#!/usr/bin/env python3
"""
Train an MLP with optional static weight sparsity.

Supported datasets:
- Fashion-MNIST
- CIFAR-10

Features:
- Dropout
- Weight decay (L2)
- Label smoothing (optional)
- MixUp (optional)
- Cosine LR schedule
- Early stopping
- Static unstructured sparsity via masked linear layers
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

from sparsity import MaskedLinear


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Dataset / Transforms
# --------------------------------------------------

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
):
    train_tf, test_tf, num_classes, in_ch, img_size = build_transforms(dataset)

    if dataset.lower() == "fashionmnist":
        full_train = torchvision.datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=train_tf
        )
        test_set = torchvision.datasets.FashionMNIST(
            data_dir, train=False, download=True, transform=test_tf
        )
    else:
        full_train = torchvision.datasets.CIFAR10(
            data_dir, train=True, download=True, transform=train_tf
        )
        test_set = torchvision.datasets.CIFAR10(
            data_dir, train=False, download=True, transform=test_tf
        )

    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size

    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=g)

    return train_set, val_set, test_set, num_classes, in_ch, img_size


# --------------------------------------------------
# MixUp (optional)
# --------------------------------------------------

def mixup_batch(x, y, alpha: float):
    if alpha <= 0.0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_loss(logits, y_a, y_b, lam, label_smoothing: float):
    return lam * F.cross_entropy(
        logits, y_a, label_smoothing=label_smoothing
    ) + (1.0 - lam) * F.cross_entropy(
        logits, y_b, label_smoothing=label_smoothing
    )


# --------------------------------------------------
# Model
# --------------------------------------------------

class MLP(nn.Module):
    """Fully-connected MLP with static weight sparsity."""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout, sparsity):
        super().__init__()

        self.net = nn.Sequential(
            MaskedLinear(input_dim, hidden_dim, sparsity=sparsity),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            MaskedLinear(hidden_dim, hidden_dim, sparsity=sparsity),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            MaskedLinear(hidden_dim, num_classes, sparsity=sparsity)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------
# Train / Eval
# --------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
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

    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n
    }


def train_one_epoch(
    model, loader, optimizer, device, label_smoothing, mixup_alpha
):
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

    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n
    }


# --------------------------------------------------
# Training State
# --------------------------------------------------

@dataclass
class BestState:
    best_val_acc: float = -1.0
    best_epoch: int = -1
    state_dict_cpu: Optional[Dict[str, torch.Tensor]] = None


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashionmnist")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    # Sparsity
    parser.add_argument("--sparsity", type=float, default=0.0,
                        help="Fraction of weights set to zero (e.g. 0.8 = 80%)")

    # Optional regularization
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)

    # Early stopping
    parser.add_argument("--early-stop-patience", type=int, default=5)

    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_set, val_set, test_set, num_classes, in_ch, img_size = load_data(
        args.dataset, args.data_dir, args.val_fraction, args.seed
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    input_dim = in_ch * img_size * img_size
    model = MLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        sparsity=args.sparsity
    ).to(device)

    print(f"Trainable params: {count_params(model):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best = BestState()
    patience = args.early_stop_patience
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, device,
            args.label_smoothing, args.mixup
        )
        val_m = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={train_m['acc']:.4f} "
            f"val_acc={val_m['acc']:.4f}"
        )

        if val_m["acc"] > best.best_val_acc:
            best.best_val_acc = val_m["acc"]
            best.best_epoch = epoch
            best.state_dict_cpu = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience = args.early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping.")
                break

    train_time = time.time() - t0

    if best.state_dict_cpu is not None:
        model.load_state_dict(best.state_dict_cpu)

    test_m = evaluate(model, test_loader, device)

    print("\n=== FINAL RESULTS ===")
    print(f"Best val acc: {best.best_val_acc:.4f}")
    print(f"Test acc:     {test_m['acc']:.4f}")
    print(f"Train time:   {train_time:.2f}s")


if __name__ == "__main__":
    main()
