#!/usr/bin/env python3
import glob
import os
import re
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def read_metrics(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def main():
    base = Path("experiments/results")
    paths = sorted(base.glob("**/metrics.csv"))

    # sparsity -> list of (best_val_acc, final_val_acc, final_grad_norm)
    agg = defaultdict(list)

    # Extract sparsity from folder name: mlp_fmnist_s0.8_seed0
    pat = re.compile(r"_s([0-9.]+)_seed")

    for p in paths:
        m = pat.search(str(p.parent))
        if not m:
            continue
        s = float(m.group(1))

        rows = read_metrics(p)
        if not rows:
            continue

        val_accs = [float(r["val_acc"]) for r in rows]
        grad_norms = [float(r["train_grad_norm"]) for r in rows]

        best_val = max(val_accs)
        final_val = val_accs[-1]
        final_gn = grad_norms[-1]

        agg[s].append((best_val, final_val, final_gn))

    if not agg:
        print("No metrics found. Did you run the sweep?")
        return

    sparsities = sorted(agg.keys())
    best_mean, best_std = [], []
    final_mean, final_std = [], []
    gn_mean, gn_std = [], []

    for s in sparsities:
        vals = np.array(agg[s], dtype=float)  # shape (runs, 3)
        best_mean.append(vals[:,0].mean()); best_std.append(vals[:,0].std())
        final_mean.append(vals[:,1].mean()); final_std.append(vals[:,1].std())
        gn_mean.append(vals[:,2].mean()); gn_std.append(vals[:,2].std())

    out_dir = Path("experiments/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: sparsity vs best val acc
    plt.figure()
    plt.errorbar(sparsities, best_mean, yerr=best_std, marker="o", capsize=3)
    plt.xlabel("Sparsity (fraction of zero weights)")
    plt.ylabel("Best validation accuracy")
    plt.title("Sparsity vs Best Val Accuracy (mean ± std over seeds)")
    plt.grid(True)
    plt.savefig(out_dir / "sparsity_vs_best_val_acc.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 2: sparsity vs final val acc
    plt.figure()
    plt.errorbar(sparsities, final_mean, yerr=final_std, marker="o", capsize=3)
    plt.xlabel("Sparsity (fraction of zero weights)")
    plt.ylabel("Final validation accuracy")
    plt.title("Sparsity vs Final Val Accuracy (mean ± std over seeds)")
    plt.grid(True)
    plt.savefig(out_dir / "sparsity_vs_final_val_acc.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 3: sparsity vs gradient norm
    plt.figure()
    plt.errorbar(sparsities, gn_mean, yerr=gn_std, marker="o", capsize=3)
    plt.xlabel("Sparsity (fraction of zero weights)")
    plt.ylabel("Train gradient norm (final epoch)")
    plt.title("Sparsity vs Gradient Norm (mean ± std over seeds)")
    plt.grid(True)
    plt.savefig(out_dir / "sparsity_vs_grad_norm.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved plots to:", out_dir)

if __name__ == "__main__":
    main()

