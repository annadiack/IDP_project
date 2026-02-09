#!/usr/bin/env python3
"""
Plot comparisons for MLP Dense vs MLP Iterative (RAM + time metrics, no FLOPs/GPU).

Expected metrics.csv columns (both runs):
epoch,lr,train_loss,train_acc,train_grad_norm,val_loss,val_acc,
prune_step,pruned_now,mask_sparsity,
epoch_time_sec,elapsed_time_sec,time_to_acc_sec,time_to_acc_epoch,
epoch_peak_rss_mb,epoch_end_rss_mb,sys_used_mb,sys_avail_mb

Usage:
  python3 plot_mlp_dense_vs_iter.py \
    --dense ../../experiments/results/mlp_dense_fullmetrics/metrics.csv \
    --iter  ../../experiments/results/mlp_iter_fullmetrics/metrics.csv \
    --outdir plots_mlp
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PLOTS = [
    ("val_acc", "Validation Accuracy", "val_acc"),
    ("train_acc", "Train Accuracy", "train_acc"),
    ("val_loss", "Validation Loss", "loss"),
    ("train_loss", "Train Loss", "loss"),
    ("train_grad_norm", "Train Grad Norm", "grad_norm"),

    ("epoch_time_sec", "Time per Epoch", "seconds"),
    ("elapsed_time_sec", "Elapsed Training Time", "seconds"),

    ("epoch_peak_rss_mb", "Peak Process RAM per Epoch (RSS)", "MB"),
    ("epoch_end_rss_mb", "End-of-Epoch Process RAM (RSS)", "MB"),

    ("mask_sparsity", "Iterative Mask Sparsity", "sparsity"),

    ("nnz_params", "Number of non-zero parameters", "nnz_params"),
]


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python")

    # Normalize headers: trim whitespace + remove BOM if present
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    if "epoch" not in df.columns:
        raise ValueError(f"{path} missing 'epoch'. Columns: {df.columns.tolist()}")

    # Convert to numeric (blank -> NaN)
    for c in df.columns:
        if c != "epoch":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("epoch").reset_index(drop=True)


def time_to_acc_summary(df: pd.DataFrame) -> str:
    if "time_to_acc_sec" not in df.columns:
        return "TTA: n/a"
    s = df["time_to_acc_sec"].dropna()
    if s.empty:
        return "TTA: not reached"
    t = float(s.iloc[0])
    e = None
    if "time_to_acc_epoch" in df.columns:
        es = df["time_to_acc_epoch"].dropna()
        if not es.empty:
            e = int(es.iloc[0])
    return f"TTA: {t:.2f}s" + (f" (epoch {e})" if e is not None else "")


def plot_metric(
    df_dense: pd.DataFrame,
    df_iter: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    outpath: Path | None,
    overlay_sparsity: bool = True,
):
    fig, ax = plt.subplots(figsize=(9, 5))

    plotted_any = False
    handles, labels = [], []

    if metric in df_dense.columns and not df_dense[metric].isna().all():
        ln, = ax.plot(df_dense["epoch"], df_dense[metric], marker="o", linewidth=2, label="Dense")
        handles.append(ln); labels.append("Dense")
        plotted_any = True

    if metric in df_iter.columns and not df_iter[metric].isna().all():
        ln, = ax.plot(df_iter["epoch"], df_iter[metric], marker="o", linewidth=2, label="Iterative")
        handles.append(ln); labels.append("Iterative")
        plotted_any = True

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # Overlay sparsity (secondary axis) for everything except the sparsity plot itself
    if overlay_sparsity and metric != "mask_sparsity" and plotted_any:
        if "mask_sparsity" in df_iter.columns and not df_iter["mask_sparsity"].isna().all():
            ax2 = ax.twinx()
            s, = ax2.plot(
                df_iter["epoch"], df_iter["mask_sparsity"],
                linestyle="--", linewidth=1.5, label="Iterative sparsity"
            )
            ax2.set_ylabel("mask_sparsity")
            ax2.set_ylim(0.0, 1.0)
            handles.append(s); labels.append("Iterative sparsity")

    if not plotted_any:
        ax.text(0.5, 0.5, f"No numeric data for '{metric}'", ha="center", va="center", transform=ax.transAxes)

    if handles:
        ax.legend(handles=handles, labels=labels, loc="best")

    fig.tight_layout()
    if outpath is None:
        plt.show()
    else:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=200)
        plt.close(fig)


def print_diagnostics(df_dense: pd.DataFrame, df_iter: pd.DataFrame):
    rows = []
    for metric, _, _ in PLOTS:
        rows.append({
            "metric": metric,
            "dense_non_nan": int(df_dense[metric].notna().sum()) if metric in df_dense.columns else 0,
            "iter_non_nan": int(df_iter[metric].notna().sum()) if metric in df_iter.columns else 0,
        })
    print("\n=== Diagnostics (non-NaN) ===")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True, help="Dense metrics.csv")
    ap.add_argument("--iter", required=True, help="Iterative metrics.csv")
    ap.add_argument("--outdir", default="", help="If set, saves PNGs; otherwise shows plots.")
    ap.add_argument("--prefix", default="mlp_dense_vs_iter", help="Filename prefix")
    ap.add_argument("--no-sparsity-overlay", action="store_true")
    args = ap.parse_args()

    df_dense = read_csv(args.dense)
    df_iter = read_csv(args.iter)

    print("Dense columns:", df_dense.columns.tolist())
    print("Iter columns:", df_iter.columns.tolist())

    print_diagnostics(df_dense, df_iter)

    subtitle = f"Dense({time_to_acc_summary(df_dense)}) vs Iter({time_to_acc_summary(df_iter)})"

    outdir = Path(args.outdir) if args.outdir else None

    def out(name: str) -> Path | None:
        if outdir is None:
            return None
        return outdir / f"{args.prefix}_{name}.png"

    overlay = not args.no_sparsity_overlay

    for metric, title, ylabel in PLOTS:
        if metric == "mask_sparsity":
            # plot sparsity alone (iter only)
            plot_metric(df_dense, df_iter, metric, f"{title}\n{subtitle}", ylabel, out(metric), overlay_sparsity=False)
        else:
            plot_metric(df_dense, df_iter, metric, f"{title}\n{subtitle}", ylabel, out(metric), overlay_sparsity=overlay)

    print("\nDone.")


if __name__ == "__main__":
    main()
