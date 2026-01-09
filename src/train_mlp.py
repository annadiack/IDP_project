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
- Static unstructured sparsity via masked linear layers (MaskedLinear)
- CSV logging (train/val metrics, grad norm, effective sparsity)
"""

import argparse
import time
import random
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

# IMPORTANT: sparsity.py must be in the same folder (src/)
from sparsity import MaskedLinear


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed Python/NumPy/PyTorch RNGs for reproducibility."""
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


# --------------------------------------------------
# Dataset / Transforms
# --------------------------------------------------

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

        # Common augmentation recipe for CIFAR-10
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
    """Download dataset and split training set into train/val (reproducibly)."""
    train_tf, test_tf, num_classes, in_ch, img_size = build_transforms(dataset)

    if dataset.lower() == "fashionmnist":
        full_train = torchvision.datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=train_tf
        )
        test_set = torchvision.datasets.FashionMNIST(
            data_dir, train=False, download=True, transform=test_tf
        )
    else:  # cifar10
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

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """
    MixUp: create convex combinations of pairs of samples.
    Returns:
      mixed_x, y_a, y_b, lam
    """
    if alpha <= 0.0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_loss(logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float, label_smoothing: float):
    """Compute MixUp loss as weighted sum of two cross-entropies."""
    return lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing) + (1.0 - lam) * F.cross_entropy(
        logits, y_b, label_smoothing=label_smoothing
    )


# --------------------------------------------------
# Model
# --------------------------------------------------

class MLP(nn.Module):
    """Fully-connected MLP with static weight sparsity (MaskedLinear)."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float, sparsity: float):
        super().__init__()

        self.net = nn.Sequential(
            MaskedLinear(input_dim, hidden_dim, sparsity=sparsity),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            MaskedLinear(hidden_dim, hidden_dim, sparsity=sparsity),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            MaskedLinear(hidden_dim, num_classes, sparsity=sparsity),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.net(x)


def count_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def count_nnz_params(model: nn.Module) -> int:
    """Count non-zero parameters across all trainable tensors."""
    nnz = 0
    for p in model.parameters():
        if p.requires_grad:
            nnz += int((p != 0).sum().item())
    return nnz


@torch.no_grad()
def effective_sparsity(model: nn.Module) -> float:
    """Fraction of zero parameters among trainable parameters."""
    total = 0
    nnz = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
            nnz += int((p != 0).sum().item())
    if total == 0:
        return 0.0
    return 1.0 - (nnz / total)


# --------------------------------------------------
# Train / Eval
# --------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Compute average loss and accuracy on loader."""
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


@torch.no_grad()
def grad_global_norm(model: nn.Module) -> float:
    """
    Global L2 norm of gradients over all trainable parameters.
    Call this AFTER loss.backward().
    """
    sq_sum = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            sq_sum += float(p.grad.detach().pow(2).sum().item())
    return float(np.sqrt(sq_sum))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float,
    mixup_alpha: float
) -> Dict[str, float]:
    """One training epoch with optional MixUp + label smoothing; returns avg grad norm too."""
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    total_grad_norm = 0.0
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Optionally apply MixUp augmentation
        if mixup_alpha > 0.0:
            x_mix, y_a, y_b, lam = mixup_batch(x, y, alpha=mixup_alpha)
            logits = model(x_mix)
            loss = mixup_loss(logits, y_a, y_b, lam, label_smoothing=label_smoothing)

            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()

        loss.backward()

        # Measure gradient norm after backward, before optimizer step
        gn = grad_global_norm(model)
        total_grad_norm += gn
        num_batches += 1

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += correct
        total_n += x.size(0)

    return {
        "loss": total_loss / max(1, total_n),
        "acc": total_correct / max(1, total_n),
        "grad_norm": total_grad_norm / max(1, num_batches),
    }


# --------------------------------------------------
# Training State
# --------------------------------------------------

@dataclass
class BestState:
    """Keep track of best validation checkpoint."""
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
                        help="Fraction of weights set to zero (e.g. 0.8 = 80%).")

    # Optional regularization
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)

    # Early stopping
    parser.add_argument("--early-stop-patience", type=int, default=5)

    # Output / logging
    parser.add_argument("--out-dir", type=str, default="../experiments/results",
                        help="Where to store metrics.csv for this run.")
    parser.add_argument("--run-name", type=str, default="mlp_run",
                        help="Subfolder name inside out-dir.")

    args = parser.parse_args()

    # Basic sanity checks
    if not (0.0 <= args.sparsity < 1.0):
        raise ValueError("--sparsity must be in [0.0, 1.0).")

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Prepare output folder
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"

    # Load data
    train_set, val_set, test_set, num_classes, in_ch, img_size = load_data(
        args.dataset, args.data_dir, args.val_fraction, args.seed
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Build model
    input_dim = in_ch * img_size * img_size
    model = MLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        sparsity=args.sparsity
    ).to(device)

    total_params = count_params(model)
    nnz = count_nnz_params(model)
    eff_s = effective_sparsity(model)

    print(f"Total params: {total_params:,} | NNZ: {nnz:,} | effective sparsity: {eff_s:.4f}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "lr",
            "train_loss", "train_acc", "train_grad_norm",
            "val_loss", "val_acc",
            "effective_sparsity"
        ])

    best = BestState()
    patience_left = args.early_stop_patience

    t0_total = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer, device,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup
        )
        val_m = evaluate(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr={lr_now:.5f} | "
            f"train_acc={train_m['acc']:.4f} val_acc={val_m['acc']:.4f} | "
            f"train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f} | "
            f"grad_norm={train_m['grad_norm']:.4f} | "
            f"epoch_time={epoch_time:.2f}s"
        )

        # Append CSV row
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, lr_now,
                train_m["loss"], train_m["acc"], train_m["grad_norm"],
                val_m["loss"], val_m["acc"],
                eff_s
            ])

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

    train_time = time.time() - t0_total

    # Restore best model and test
    if best.state_dict_cpu is not None:
        model.load_state_dict(best.state_dict_cpu)

    test_m = evaluate(model, test_loader, device)

    print("\n=== FINAL (Best Val Checkpoint) ===")
    print(f"Best val acc: {best.best_val_acc:.4f} at epoch {best.best_epoch}")
    print(f"Test acc:     {test_m['acc']:.4f}")
    print(f"Train time:   {train_time:.2f} seconds")
    print(f"Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
