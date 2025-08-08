#!/usr/bin/env python3
"""sweep.py
Lightweight overnight grid runner for ft.py. Sequential by default (safe on one GPU).
Edit the GRIDS below to fit your budget. Uses subprocess to spawn child runs.

Usage
-----
python sweep.py \
  --train_file train.jsonl --val_file val.jsonl \
  --text1_column edu1 --text2_column edu2 --label_column label \
  --wandb_project EDU-Relations --wandb_entity your-team --wandb_group overnight

Tips
----
- Set CUDA_VISIBLE_DEVICES before running if you want to pin a GPU.
- Use --dry_run to just print commands.
- Use --max_runs to cap total combinations.
"""
import argparse
import itertools
import os
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file", required=True)
    p.add_argument("--text1_column", default="edu1")
    p.add_argument("--text2_column", default="edu2")
    p.add_argument("--label_column", default="label")

    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_group", default=None)

    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_runs", type=int, default=None)
    p.add_argument("--base_outdir", default="runs")

    return p.parse_args()


# -----------------------
# Define your search grid
# -----------------------
GRIDS = {
    "encoding_mode": ["literal_sep_maxlen", "tokenizer_sep_dynamic"],
    "batch_size": [64, 128],
    "lr": [2e-5, 3e-5, 4e-5],
    "label_smoothing": [0.0, 0.05],
    "rebuttal_weight_scale": [0.9, 1.0, 1.1],
    "epochs": [40],  # keep long enough to reach plateau
}

# You can add more: weight_decay, patience (via code edit), etc.


def main():
    args = parse_args()

    keys = list(GRIDS.keys())
    values = [GRIDS[k] for k in keys]
    combos = list(itertools.product(*values))

    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.base_outdir) / f"sweep_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    for i, combo in enumerate(combos, start=1):
        cfg = dict(zip(keys, combo))
        tag = "_".join(f"{k}-{str(v).replace('.', '')}" for k, v in cfg.items())
        outdir = out_root / tag
        outdir.mkdir(parents=True, exist_ok=True)

        run_name = f"{cfg['encoding_mode']}_bs{cfg['batch_size']}_lr{cfg['lr']}_ls{cfg['label_smoothing']}_rws{cfg['rebuttal_weight_scale']}"

        cmd = [
            "python", "ft.py",
            "--train_file", args.train_file,
            "--val_file", args.val_file,
            "--text1_column", args.text1_column,
            "--text2_column", args.text2_column,
            "--label_column", args.label_column,
            "--encoding_mode", cfg["encoding_mode"],
            "--batch_size", str(cfg["batch_size"]),
            "--epochs", str(cfg["epochs"]),
            "--lr", str(cfg["lr"]),
            "--label_smoothing", str(cfg["label_smoothing"]),
            "--rebuttal_weight_scale", str(cfg["rebuttal_weight_scale"]),
            "--output_dir", str(outdir),
            "--run_name", run_name,
        ]

        if args.wandb_project:
            cmd += ["--wandb_project", args.wandb_project]
        if args.wandb_entity:
            cmd += ["--wandb_entity", args.wandb_entity]
        if args.wandb_group:
            cmd += ["--wandb_group", args.wandb_group]

        print(f"\n[{i}/{len(combos)}] →", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
