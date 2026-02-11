#!/usr/bin/env python3
"""
train_mlp_dense_metrics.py

Dense MLP training (Fashion-MNIST / CIFAR-10) with M1/MPS-friendly metrics.

Writes metrics.csv with:
epoch,lr,train_loss,train_acc,train_grad_norm,val_loss,val_acc,
prune_step,pruned_now,mask_sparsity,
epoch_time_sec,elapsed_time_sec,time_to_acc_sec,time_to_acc_epoch,
epoch_peak_rss_mb,epoch_end_rss_mb,sys_used_mb,sys_avail_mb,
total_params,nnz_params,nnz_ratio
"""

import argparse
import time
import random
import csv
import os
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

import psutil


# -------------------------
# RAM helpers (TOTAL RSS)
# -------------------------
_MAIN_PROC = psutil.Process(os.getpid())

def rss_mb_total(include_children: bool = True) -> float:
    total = _MAIN_PROC.memory_info().rss
    if include_children:
        for ch in _MAIN_PROC.children(recursive=True):
            try:
                total += ch.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return total / (1024.0 * 1024.0)

def sys_used_mb() -> float:
    return psutil.virtual_memory().used / (1024.0 * 1024.0)

def sys_avail_mb() -> float:
    return psutil.virtual_memory().available / (1024.0 * 1024.0)


# -------------------------
# Reproducibility / device
# -------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Dataset / Transforms
# -------------------------
def build_transforms(dataset: str) -> Tuple[T.Compose, T.Compose, int, int, int]:
    dataset = dataset.lower()

    if dataset == "fashionmnist":
        mean, std = (0.2860,), (0.3530,)
        tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        return tf, tf, 10, 1, 28

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


def load_data(dataset: str, data_dir: str, val_fraction: float, seed: int):
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


# -------------------------
# MixUp (optional)
# -------------------------
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def mixup_loss(logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float, label_smoothing: float):
    return lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing) + (1.0 - lam) * F.cross_entropy(
        logits, y_b, label_smoothing=label_smoothing
    )


# -------------------------
# Model (dense)
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.net(x)


# -------------------------
# Param counting (per epoch)
# -------------------------
@torch.no_grad()
def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def count_nnz_params(model: nn.Module, include_bias: bool = True) -> int:
    nnz = 0
    # If any module exposes a binary mask, count from it (not used in dense, but keeps API consistent)
    for m in model.modules():
        if hasattr(m, "mask") and isinstance(getattr(m, "mask"), torch.Tensor) and hasattr(m, "weight"):
            nnz += int(m.mask.sum().item())
            if include_bias and getattr(m, "bias", None) is not None:
                nnz += m.bias.numel()
    if nnz > 0:
        return nnz

    # Dense fallback
    for p in model.parameters():
        if p.requires_grad:
            if (not include_bias) and p.ndim == 1:
                continue
            nnz += int((p != 0).sum().item())
    return nnz


# -------------------------
# Train/Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
    return {"loss": total_loss / max(1, total_n), "acc": total_correct / max(1, total_n)}


@torch.no_grad()
def grad_global_norm(model: nn.Module) -> float:
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
    mixup_alpha: float,
    include_children_rss: bool,
) -> Dict[str, float]:
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    total_grad_norm = 0.0
    num_batches = 0

    epoch_peak_rss = rss_mb_total(include_children=include_children_rss)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if mixup_alpha > 0.0:
            x_mix, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
            logits = model(x_mix)
            loss = mixup_loss(logits, y_a, y_b, lam, label_smoothing=label_smoothing)
            correct = (logits.argmax(1) == y).sum().item()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            correct = (logits.argmax(1) == y).sum().item()

        loss.backward()

        total_grad_norm += grad_global_norm(model)
        num_batches += 1

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += correct
        total_n += x.size(0)

        epoch_peak_rss = max(epoch_peak_rss, rss_mb_total(include_children=include_children_rss))

    return {
        "loss": total_loss / max(1, total_n),
        "acc": total_correct / max(1, total_n),
        "grad_norm": total_grad_norm / max(1, num_batches),
        "epoch_peak_rss_mb": epoch_peak_rss,
        "epoch_end_rss_mb": rss_mb_total(include_children=include_children_rss),
        "sys_used_mb": sys_used_mb(),
        "sys_avail_mb": sys_avail_mb(),
    }


# -------------------------
# Best checkpoint
# -------------------------
@dataclass
class BestState:
    best_val_acc: float = -1.0
    best_epoch: int = -1
    state_dict_cpu: Optional[Dict[str, torch.Tensor]] = None


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="fashionmnist")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=5)

    p.add_argument("--tta-acc", type=float, default=0.90)

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--include-children-rss", action="store_true", default=True)

    p.add_argument("--out-dir", type=str, default="../experiments/results")
    p.add_argument("--run-name", type=str, default="mlp_dense_fullmetrics")
    args = p.parse_args()

    if args.mixup != 0.0 or args.label_smoothing != 0.0:
        print("Mixup and label smoothing are disabled for this script; forcing both to 0.0.")
    args.mixup = 0.0
    args.label_smoothing = 0.0

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"

    train_set, val_set, test_set, num_classes, in_ch, img_size = load_data(
        args.dataset, args.data_dir, args.val_fraction, args.seed
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_dim = in_ch * img_size * img_size
    model = MLP(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=num_classes, dropout=args.dropout).to(device)

    total_params_const = count_total_params(model)
    print(f"Total params: {total_params_const:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # CSV header
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "lr",
            "train_loss", "train_acc", "train_grad_norm",
            "val_loss", "val_acc",
            "prune_step", "pruned_now", "mask_sparsity",
            "epoch_time_sec", "elapsed_time_sec",
            "time_to_acc_sec", "time_to_acc_epoch",
            "epoch_peak_rss_mb", "epoch_end_rss_mb", "sys_used_mb", "sys_avail_mb",
            "total_params", "nnz_params", "nnz_ratio",
        ])

    best = BestState()
    patience_left = args.early_stop_patience

    t0_total = time.perf_counter()
    time_to_acc_sec: Optional[float] = None
    time_to_acc_epoch: Optional[int] = None

    for epoch in range(1, args.epochs + 1):
        t0_epoch = time.perf_counter()

        train_m = train_one_epoch(
            model, train_loader, optimizer, device,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup,
            include_children_rss=args.include_children_rss,
        )
        val_m = evaluate(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        epoch_time_sec = float(time.perf_counter() - t0_epoch)
        elapsed_time_sec = float(time.perf_counter() - t0_total)

        if time_to_acc_sec is None and val_m["acc"] >= args.tta_acc:
            time_to_acc_sec = elapsed_time_sec
            time_to_acc_epoch = epoch

        # Dense baseline pruning fields
        prune_step, pruned_now, mask_sparsity = 0, 0.0, 0.0

        # Params per epoch
        total_params = total_params_const
        nnz_params = count_nnz_params(model, include_bias=True)
        nnz_ratio = (nnz_params / total_params) if total_params > 0 else 0.0

        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr={lr_now:.5f} | "
            f"train_acc={train_m['acc']:.4f} val_acc={val_m['acc']:.4f} | "
            f"epoch_time={epoch_time_sec:.2f}s | peak_rss={train_m['epoch_peak_rss_mb']:.1f}MB | "
            f"nnz={nnz_ratio:.3f}"
        )

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, lr_now,
                train_m["loss"], train_m["acc"], train_m["grad_norm"],
                val_m["loss"], val_m["acc"],
                prune_step, pruned_now, mask_sparsity,
                epoch_time_sec, elapsed_time_sec,
                "" if time_to_acc_sec is None else time_to_acc_sec,
                "" if time_to_acc_epoch is None else time_to_acc_epoch,
                train_m["epoch_peak_rss_mb"], train_m["epoch_end_rss_mb"],
                train_m["sys_used_mb"], train_m["sys_avail_mb"],
                total_params, nnz_params, nnz_ratio,
            ])

        if val_m["acc"] > best.best_val_acc:
            best.best_val_acc = val_m["acc"]
            best.best_epoch = epoch
            best.state_dict_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best.state_dict_cpu is not None:
        model.load_state_dict(best.state_dict_cpu)

    test_m = evaluate(model, test_loader, device)

    print("\n=== FINAL (Best Val Checkpoint) ===")
    print(f"Best val acc: {best.best_val_acc:.4f} at epoch {best.best_epoch}")
    print(f"Test acc:     {test_m['acc']:.4f}")
    print(f"Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
