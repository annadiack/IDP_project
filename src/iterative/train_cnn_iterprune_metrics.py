#!/usr/bin/env python3
"""
train_cnn_iterprune_metrics.py

Iterative-pruning CNN training (Fashion-MNIST / CIFAR-10) with M1/MPS-friendly metrics.

Same metrics.csv schema as dense CNN + pruning columns filled.
Includes MPS-safe global pruning: computes threshold on CPU (kthvalue not supported on MPS).

No FLOPs, no GPU metrics. RAM uses psutil (RSS + system used/available).
"""

import argparse
import time
import random
import csv
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

import psutil


# -------------------------
# RAM helpers
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
# Data
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
# Prunable layers + pruning utils (self-contained)
# -------------------------
class PrunableConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class PrunableLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

def prunable_modules(model: nn.Module) -> List[nn.Module]:
    mods = []
    for m in model.modules():
        if isinstance(m, (PrunableConv2d, PrunableLinear)):
            mods.append(m)
    return mods

@torch.no_grad()
def apply_all_masks_(model: nn.Module) -> None:
    for m in prunable_modules(model):
        m.weight.mul_(m.mask)

@torch.no_grad()
def global_sparsity(model: nn.Module) -> float:
    total = 0
    nnz = 0
    for m in prunable_modules(model):
        total += m.mask.numel()
        nnz += int(m.mask.sum().item())
    if total == 0:
        return 0.0
    return 1.0 - (nnz / total)

class IterativePruneSchedule:
    def __init__(self, final_sparsity: float, steps: int):
        self.final_sparsity = final_sparsity
        self.steps = steps

    def fraction_per_step(self) -> float:
        # prune same fraction of remaining each step to reach final sparsity
        # remaining after s steps: (1-p)^s = 1-final_sparsity
        # p = 1 - (1-final)^(1/steps)
        return 1.0 - (1.0 - self.final_sparsity) ** (1.0 / self.steps)

@torch.no_grad()
def global_magnitude_prune_(model: nn.Module, prune_fraction_of_remaining: float) -> Dict[str, float]:
    """
    Prune a fraction of currently-unpruned weights globally by magnitude.
    MPS-safe: threshold computed on CPU.
    """
    if prune_fraction_of_remaining <= 0.0:
        return {"pruned_now": 0.0}

    mags = []
    for m in prunable_modules(model):
        w = m.weight.detach()
        mask = m.mask.detach()
        mags.append(w.abs()[mask.bool()].reshape(-1))
    if not mags:
        return {"pruned_now": 0.0}

    all_mags = torch.cat(mags, dim=0)
    n = all_mags.numel()
    if n == 0:
        return {"pruned_now": 0.0}

    k = int(prune_fraction_of_remaining * n)
    k = max(1, min(k, n))

    # kthvalue not supported on MPS -> do on CPU
    thresh = torch.kthvalue(all_mags.to("cpu"), k).values.item()

    pruned_count = 0
    remaining_before = 0

    for m in prunable_modules(model):
        w = m.weight.detach()
        mask = m.mask
        alive = mask.bool()
        remaining_before += int(alive.sum().item())

        to_prune = alive & (w.abs() <= thresh)
        if to_prune.any():
            mask[to_prune] = 0.0
            pruned_count += int(to_prune.sum().item())

    apply_all_masks_(model)
    pruned_now = 0.0 if remaining_before == 0 else (pruned_count / remaining_before)
    return {"pruned_now": float(pruned_now)}


# -------------------------
# Model (prunable CNN)
# -------------------------
class PrunableCNN(nn.Module):
    def __init__(self, in_ch: int, img_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.conv1 = PrunableConv2d(in_ch, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = PrunableConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, img_size, img_size)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy)), inplace=False))
            x = self.pool(F.relu(self.bn2(self.conv2(x)), inplace=False))
            feat_dim = x.flatten(1).shape[1]

        self.fc1 = PrunableLinear(feat_dim, 128)
        self.fc2 = PrunableLinear(128, num_classes)

        apply_all_masks_(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.pool(F.relu(self.bn2(self.conv2(x)), inplace=True))
        x = torch.flatten(x, start_dim=1)
        x = self.drop(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.drop(x)
        return self.fc2(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    epoch_peak_rss = rss_mb_total(include_children_rss)

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

        apply_all_masks_(model)

        total_loss += loss.item() * x.size(0)
        total_correct += correct
        total_n += x.size(0)

        epoch_peak_rss = max(epoch_peak_rss, rss_mb_total(include_children_rss))

    return {
        "loss": total_loss / max(1, total_n),
        "acc": total_correct / max(1, total_n),
        "grad_norm": total_grad_norm / max(1, num_batches),
        "epoch_peak_rss_mb": epoch_peak_rss,
        "epoch_end_rss_mb": rss_mb_total(include_children_rss),
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
    p.add_argument("--dataset", choices=["fashionmnist", "cifar10"], default="fashionmnist")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=7)

    p.add_argument("--tta-acc", type=float, default=0.90)

    # iterative pruning
    p.add_argument("--final-sparsity", type=float, default=0.9)
    p.add_argument("--prune-steps", type=int, default=10)
    p.add_argument("--prune-every", type=int, default=1)
    p.add_argument("--prune-warmup", type=int, default=1)

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--include-children-rss", action="store_true", default=True)

    p.add_argument("--out-dir", type=str, default="../experiments/results")
    p.add_argument("--run-name", type=str, default="cnn_iter_fullmetrics")
    args = p.parse_args()

    if not (0.0 <= args.final_sparsity < 1.0):
        raise ValueError("--final-sparsity must be in [0,1).")
    if args.prune_steps < 1:
        raise ValueError("--prune-steps must be >= 1.")
    if args.prune_every < 1:
        raise ValueError("--prune-every must be >= 1.")

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

    model = PrunableCNN(in_ch=in_ch, img_size=img_size, num_classes=num_classes, dropout=args.dropout).to(device)
    apply_all_masks_(model)
    print(f"Params: {count_params(model):,} | initial sparsity={global_sparsity(model):.4f}")

    schedule = IterativePruneSchedule(final_sparsity=args.final_sparsity, steps=args.prune_steps)
    prune_frac = schedule.fraction_per_step()
    print(f"Iter-prune: target={args.final_sparsity:.2f}, steps={args.prune_steps}, fraction/step={prune_frac:.6f}")

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
        ])

    best = BestState()
    patience_left = args.early_stop_patience

    t0_total = time.perf_counter()
    time_to_acc_sec: Optional[float] = None
    time_to_acc_epoch: Optional[int] = None

    warmup_epoch = max(1, args.prune_warmup)

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

        # pruning
        prune_step = 0
        pruned_now = 0.0
        is_after_warmup = epoch >= warmup_epoch
        is_prune_epoch = is_after_warmup and ((epoch - warmup_epoch) % args.prune_every == 0)

        if is_prune_epoch:
            prune_step = ((epoch - warmup_epoch) // args.prune_every) + 1
            if prune_step <= args.prune_steps:
                stats = global_magnitude_prune_(model, prune_fraction_of_remaining=prune_frac)
                pruned_now = float(stats.get("pruned_now", 0.0))

        mask_s = float(global_sparsity(model))

        epoch_time_sec = float(time.perf_counter() - t0_epoch)
        elapsed_time_sec = float(time.perf_counter() - t0_total)

        if time_to_acc_sec is None and val_m["acc"] >= args.tta_acc:
            time_to_acc_sec = elapsed_time_sec
            time_to_acc_epoch = epoch

        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr={lr_now:.5f} | "
            f"train_acc={train_m['acc']:.4f} val_acc={val_m['acc']:.4f} | "
            f"sparsity={mask_s:.4f} | epoch_time={epoch_time_sec:.2f}s | peak_rss={train_m['epoch_peak_rss_mb']:.1f}MB"
            + (f" | prune_step={prune_step} pruned_now={pruned_now:.4f}" if prune_step > 0 else "")
        )

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, lr_now,
                train_m["loss"], train_m["acc"], train_m["grad_norm"],
                val_m["loss"], val_m["acc"],
                prune_step, pruned_now, mask_s,
                epoch_time_sec, elapsed_time_sec,
                "" if time_to_acc_sec is None else time_to_acc_sec,
                "" if time_to_acc_epoch is None else time_to_acc_epoch,
                train_m["epoch_peak_rss_mb"], train_m["epoch_end_rss_mb"],
                train_m["sys_used_mb"], train_m["sys_avail_mb"],
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
