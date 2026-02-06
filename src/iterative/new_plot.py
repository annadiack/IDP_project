#!/usr/bin/env python3
"""
Plot MLP runs (dense, static, iterative) on the same graph.

Expected CSV format (yours):
epoch,lr,train_loss,train_acc,train_grad_norm,val_loss,val_acc,prune_step,pruned_now,mask_sparsity

Dense/static may omit the last columns; this script handles that.

Usage:
  python3 plot_mlp_runs.py \
    --dense path/to/dense/metrics.csv \
    --static path/to/static/metrics.csv \
    --iter path/to/iter/metrics.csv \
    --out mlp_compare.png
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def read_run(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        raise ValueError(f"{csv_path} has no 'epoch' column.")
    # Ensure sorted by epoch
    df = df.sort_values("epoch").reset_index(drop=True)

    # Make missing columns safe
    for col in ["train_acc", "val_acc", "train_loss", "val_loss", "mask_sparsity"]:
        if col not in df.columns:
            df[col] = None
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dense", required=True, help="Dense MLP metrics.csv")
    p.add_argument("--static", required=True, help="Static sparse MLP metrics.csv")
    p.add_argument("--iter", required=True, help="Iterative pruning MLP metrics.csv")
    p.add_argument("--title", default="MLP Comparison (Dense vs Static vs Iterative)")
    p.add_argument("--out", default="", help="Optional output image path (png/pdf). If empty, just show.")
    p.add_argument("--plot", choices=["val_acc", "train_acc", "val_loss", "train_loss"], default="val_acc")
    p.add_argument("--show-sparsity", action="store_true", help="Add mask_sparsity as a secondary y-axis (if present).")
    args = p.parse_args()

    runs = {
        "Dense": read_run(args.dense),
        "Static": read_run(args.static),
        "Iterative": read_run(args.iter),
    }

    metric = args.plot

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, df in runs.items():
        if df[metric].isna().all():
            continue
        ax.plot(df["epoch"], df[metric], marker="o", linewidth=2, label=label)

    ax.set_title(args.title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Optional secondary axis for sparsity (useful mainly for iterative/static)
    if args.show_sparsity:
        ax2 = ax.twinx()
        plotted_any = False
        for label, df in runs.items():
            if "mask_sparsity" in df.columns and not df["mask_sparsity"].isna().all():
                ax2.plot(df["epoch"], df["mask_sparsity"], linestyle="--", linewidth=1.5, label=f"{label} sparsity")
                plotted_any = True
        if plotted_any:
            ax2.set_ylabel("mask_sparsity")
            ax2.set_ylim(0.0, 1.0)

    fig.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
