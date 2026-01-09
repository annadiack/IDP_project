#!/usr/bin/env python3
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


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

    # sparsity -> list of dicts per run
    agg = defaultdict(list)

    # folder name pattern: mlp_fmnist_s0.8_seed0
    pat = re.compile(r"_s([0-9.]+)_seed")

    for p in paths:
        m = pat.search(str(p.parent))
        if not m:
            continue
        sparsity = float(m.group(1))

        rows = read_metrics(p)
        if not rows:
            continue

        val_accs = np.array([float(r["val_acc"]) for r in rows])
        grad_norms = np.array([float(r["train_grad_norm"]) for r in rows])
        eff_s = float(rows[0]["effective_sparsity"])

        agg[sparsity].append({
            "best_val_acc": float(val_accs.max()),
            "final_val_acc": float(val_accs[-1]),
            "final_grad_norm": float(grad_norms[-1]),
            "effective_sparsity": eff_s,
        })

    summary = {}

    for s, runs in agg.items():
        best_vals = np.array([r["best_val_acc"] for r in runs])
        final_vals = np.array([r["final_val_acc"] for r in runs])
        grad_vals = np.array([r["final_grad_norm"] for r in runs])
        eff_s_vals = np.array([r["effective_sparsity"] for r in runs])

        summary[str(s)] = {
            "num_runs": len(runs),
            "effective_sparsity_mean": float(eff_s_vals.mean()),
            "best_val_acc_mean": float(best_vals.mean()),
            "best_val_acc_std": float(best_vals.std()),
            "final_val_acc_mean": float(final_vals.mean()),
            "final_val_acc_std": float(final_vals.std()),
            "final_grad_norm_mean": float(grad_vals.mean()),
            "final_grad_norm_std": float(grad_vals.std()),
        }

    out_path = base / "results_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved results summary to: {out_path}")


if __name__ == "__main__":
    main()

