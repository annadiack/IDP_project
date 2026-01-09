#!/usr/bin/env python3
import subprocess
from pathlib import Path

def run(cmd):
    print("\n>>> " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    sparsities = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]
    seeds = [0, 1, 2]   # 3 seeds = stabilere Ergebnisse

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in sparsities:
        for seed in seeds:
            run_name = f"mlp_fmnist_s{s}_seed{seed}"
            cmd = [
                "python", "src/train_mlp.py",
                "--dataset", "fashionmnist",
                "--data-dir", "./data",
                "--epochs", "10",
                "--sparsity", str(s),
                "--seed", str(seed),
                "--mixup", "0.0",
                "--label-smoothing", "0.0",
                "--out-dir", str(out_dir),
                "--run-name", run_name
            ]
            run(cmd)

if __name__ == "__main__":
    main()

