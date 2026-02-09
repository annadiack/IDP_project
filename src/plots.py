#!/usr/bin/env python3
"""
Plot metrics from 4 CSV runs:
- fashion_mnist_dense
- fashion_mnist_sparse (iterative pruning)
- cifar10_dense
- cifar10_sparse (iterative pruning)

Styling:
- CIFAR-10: green
- Fashion-MNIST: orange
- Dense params: dashed
- Sparse params: solid
- Sparsity line: grey (right y-axis)

Outputs:
1) Validation accuracy
2) Validation loss
3) Params + sparsity (two subplots: Fashion-MNIST | CIFAR-10)
4) Elapsed training time
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt


# ---------- styling ----------
COLOR = {
    "cifar10": "green",
    "fashion_mnist": "orange",
}
GREY = "grey"
LINESTYLE = {"dense": "--", "sparse": "-"}


@dataclass(frozen=True)
class RunSpec:
    path: str
    dataset: Literal["cifar10", "fashion_mnist"]
    regime: Literal["dense", "sparse"]
    label: str


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"{path}: missing 'epoch'")
    return df.sort_values("epoch").reset_index(drop=True)


def _get_x(df: pd.DataFrame, x_col: str) -> pd.Series:
    if x_col not in df.columns:
        raise ValueError(f"Missing x column '{x_col}'")
    return df[x_col]


# ---------- generic plot ----------
def _plot_metric(
    runs: list[RunSpec],
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    outpath: str,
) -> None:
    plt.figure(figsize=(10, 5))

    for r in runs:
        df = _read_csv(r.path)
        if y_col not in df.columns:
            raise ValueError(f"{r.path}: missing '{y_col}'")
        x = _get_x(df, x_col)

        plt.plot(
            x,
            df[y_col],
            color=COLOR[r.dataset],
            linestyle=LINESTYLE[r.regime],
            linewidth=2.0,
            label=r.label,
        )

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


# ---------- params + sparsity (split) ----------
def _plot_params_and_sparsity_split(
    runs: list[RunSpec],
    x_col: str,
    outpath: str,
) -> None:
    """
    Two subplots:
      Left  - Fashion-MNIST
      Right - CIFAR-10

    Each subplot:
      - Left y-axis: nnz_params (dense dashed + sparse solid)
      - Right y-axis: remaining ratio (nnz_ratio) in grey (NOT mask_sparsity),
        so dense shows ~1.0 and sparse decreases as pruning removes weights.

    Also: two separate legends per subplot (one for params, one for sparsity).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for ax, dataset in zip(axes, ["fashion_mnist", "cifar10"]):
        ax2 = ax.twinx()

        dense_run = next(r for r in runs if r.dataset == dataset and r.regime == "dense")
        sparse_run = next(r for r in runs if r.dataset == dataset and r.regime == "sparse")

        df_dense = _read_csv(dense_run.path)
        df_sparse = _read_csv(sparse_run.path)

        # columns needed
        for need, df, path in [
            ("nnz_params", df_dense, dense_run.path),
            ("nnz_params", df_sparse, sparse_run.path),
            ("mask_sparsity", df_sparse, sparse_run.path),
        ]:
            if need not in df.columns:
                raise ValueError(f"{path}: missing '{need}'")

        x_dense = _get_x(df_dense, x_col)
        x_sparse = _get_x(df_sparse, x_col)

        # --- params lines (left axis) ---
        l_dense, = ax.plot(
            x_dense,
            df_dense["nnz_params"],
            color=COLOR[dataset],
            linestyle=LINESTYLE["dense"],
            linewidth=2.0,
            label="Dense params",
        )
        l_sparse, = ax.plot(
            x_sparse,
            df_sparse["nnz_params"],
            color=COLOR[dataset],
            linestyle=LINESTYLE["sparse"],
            linewidth=2.0,
            label="Sparse params",
        )

        # --- remaining ratio lines (right axis) ---
        # This fixes the issue: dense should be ~1.0 (100% remaining).
        l_sparse_r, = ax2.plot(
            x_sparse,
            df_sparse["mask_sparsity"],
            color=GREY,
            linestyle="-",
            linewidth=2.0,
            alpha=0.9,
            label="Sparseity (mask_sparsity)",
        )

        ax.set_title("Fashion-MNIST" if dataset == "fashion_mnist" else "CIFAR-10")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Non-zero parameters (nnz_params)")
        ax2.set_ylabel("Remaining weights ratio (mask_sparsity)")
        ax2.set_ylim(0.0, 1.0)

        ax.grid(True, alpha=0.3)

        # --- two legends per subplot ---
        # Legend 1: params (left axis)
        leg1 = ax.legend(handles=[l_dense, l_sparse, l_sparse_r], loc="upper left", title="Parameters")
        ax.add_artist(leg1)

    fig.suptitle("Model Size and Remaining-Weights Ratio During Training", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fashion-dense", required=True)
    ap.add_argument("--fashion-sparse", required=True)
    ap.add_argument("--cifar-dense", required=True)
    ap.add_argument("--cifar-sparse", required=True)
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--x", default="epoch", choices=["epoch", "elapsed_time_sec"])
    args = ap.parse_args()

    runs = [
        RunSpec(args.fashion_dense, "fashion_mnist", "dense", "Fashion-MNIST (dense)"),
        RunSpec(args.fashion_sparse, "fashion_mnist", "sparse", "Fashion-MNIST (sparse)"),
        RunSpec(args.cifar_dense, "cifar10", "dense", "CIFAR-10 (dense)"),
        RunSpec(args.cifar_sparse, "cifar10", "sparse", "CIFAR-10 (sparse)"),
    ]

    outdir = args.outdir.rstrip("/")

    _plot_metric(
        runs, args.x, "val_acc",
        "Validation Accuracy", "val_acc",
        os.path.join(outdir, "val_accuracy.png")
    )

    _plot_metric(
        runs, args.x, "val_loss",
        "Validation Loss", "val_loss",
        os.path.join(outdir, "val_loss.png")
    )

    _plot_params_and_sparsity_split(
        runs,
        args.x,
        os.path.join(outdir, "params_and_sparsity_split.png"),
    )

    _plot_metric(
        runs, args.x, "elapsed_time_sec",
        "Elapsed Training Time", "elapsed_time_sec",
        os.path.join(outdir, "elapsed_time.png")
    )

    print(f"Plots saved to {outdir}/")


if __name__ == "__main__":
    main()
