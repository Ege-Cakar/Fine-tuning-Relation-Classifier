#!/usr/bin/env python3

"""
Shuffle and split a JSONL dataset into train/val/test splits with reproducibility.

Usage example:

  python /workspace/synthetic_preprocess.py \
    --input /workspace/synthetic_edu_pairs_cleaned.jsonl \
    --output-dir /workspace \
    --seed 42 \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 

Outputs (by default):
  - synthetic_edu_pairs_train.jsonl
  - synthetic_edu_pairs_val.jsonl
  - synthetic_edu_pairs_test.jsonl
in the specified output directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Iterable, List, Dict, Any


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    records: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> None:
    """Write an iterable of dictionaries to a JSONL file."""
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_split_sizes(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """Compute split sizes using floor for train/val and remainder for test."""
    if total < 0:
        raise ValueError("Total number of examples cannot be negative")

    if any(r < 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Ratios must be non-negative")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not (0.999 <= ratio_sum <= 1.001):
        raise ValueError(
            f"Ratios must sum to 1.0 (±0.001). Got {ratio_sum:.6f} instead."
        )

    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return train_size, val_size, test_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shuffle and split JSONL dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/workspace/synthetic_edu_pairs_cleaned.jsonl"),
        help="Path to the input JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace"),
        help="Directory to write the split JSONL files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation split",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for test split",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="synthetic_edu_pairs",
        help="Output filename prefix for the split files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir
    seed: int = args.seed
    train_ratio: float = args.train_ratio
    val_ratio: float = args.val_ratio
    test_ratio: float = args.test_ratio
    prefix: str = args.prefix

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    examples = read_jsonl(input_path)

    # Reproducible shuffle
    rng = random.Random(seed)
    rng.shuffle(examples)

    train_size, val_size, test_size = compute_split_sizes(
        total=len(examples),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_examples = examples[:train_size]
    val_examples = examples[train_size : train_size + val_size]
    test_examples = examples[train_size + val_size :]

    train_path = output_dir / f"{prefix}_train.jsonl"
    val_path = output_dir / f"{prefix}_val.jsonl"
    test_path = output_dir / f"{prefix}_test.jsonl"

    write_jsonl(train_examples, train_path)
    write_jsonl(val_examples, val_path)
    write_jsonl(test_examples, test_path)

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output_dir": str(output_dir),
                "seed": seed,
                "sizes": {
                    "total": len(examples),
                    "train": len(train_examples),
                    "val": len(val_examples),
                    "test": len(test_examples),
                },
                "files": {
                    "train": str(train_path),
                    "val": str(val_path),
                    "test": str(test_path),
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()


