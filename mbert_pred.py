#!/usr/bin/env python
"""modernbert_predict.py

Standalone **batch‑inference** script for checkpoints produced by the prompt‑style
`modernbert_finetune.py` (where each example was formatted as:

```
Argument 1: <EDU1>, Argument 2: <EDU2>, Relationship:
```
)

Example
-------
```bash
python modernbert_predict.py \
  --model_dir ./modbert_output \
  --input_file datasets/test.jsonl \
  --text1_column edu1 --text2_column edu2 \
  --batch_size 64
```

The script auto‑detects GPU, uses FP16 if possible, and writes one JSON line per
input containing the predicted `label` and the max‑probability `score`.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.nn.functional import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import shutil

# ---------------------------------------------------------------------
#                               CLI                                     
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Batch inference for ModernBERT checkpoints (prompt style)")
    p.add_argument("--model_dir", required=True, help="Fine‑tuned model directory")
    p.add_argument("--input_file", required=True, help="JSONL/CSV/TXT file with EDU columns")
    p.add_argument("--output_file", default=None, help="Destination JSONL; default=<model_dir>/predictions.jsonl")
    p.add_argument("--text1_column", default="edu1", help="Name of first EDU column")
    p.add_argument("--text2_column", default="edu2", help="Name of second EDU column")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    return p.parse_args()

# ---------------------------------------------------------------------
#                          helpers                                      
# ---------------------------------------------------------------------

def load_texts(path: str, col1: str, col2: str):
    ext = Path(path).suffix.lower()
    if ext in {".csv", ".tsv"}:
        ds = load_dataset("csv", data_files={"data": path})["data"]
    elif ext in {".json", ".jsonl"}:
        ds = load_dataset("json", data_files={"data": path})["data"]
    else:  # treat as plain .txt (one JSON string containing both cols is unlikely)
        ds = load_dataset("text", data_files={"data": path})["data"]
        col1, col2 = "text", None  # will error out later if second col is expected

    # sanity
    for c in (col1, col2):
        if c and c not in ds.column_names:
            raise ValueError(f"Column '{c}' not found in {path}. Available: {ds.column_names}")
    return ds, col1, col2


def add_prompt(batch: Dict[str, List[str]], col1: str, col2: str):
    prompts = [
        f"Argument 1: {e1}, Argument 2: {e2}, Relationship:"
        for e1, e2 in zip(batch[col1], batch[col2])
    ]
    return {"text": prompts}

# ---------------------------------------------------------------------
#                            main                                       
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- load model & tokenizer ----------
    model_path = args.model_dir
    best_sub = os.path.join(model_path, "best_model")
    if os.path.isdir(best_sub):            # prefer the curated best checkpoint
        model_path = best_sub
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device).eval()

    # label mapping
    with open(os.path.join(args.model_dir, "label2id.json")) as f:
        id2label = {v: k for k, v in json.load(f).items()}

    # ---------- load dataset ----------
    ds, col1, col2 = load_texts(args.input_file, args.text1_column, args.text2_column)

    # create prompt → "text" + drop raw EDU columns
    ds = ds.map(lambda b: add_prompt(b, col1, col2), batched=True, remove_columns=[c for c in (col1, col2) if c])

    # tokenise
    ds = ds.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=args.max_length),
                batched=True, remove_columns=["text"])

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         collate_fn=DataCollatorWithPadding(tokenizer))

    preds: List[Dict[str, str]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits
            probs = softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            preds.extend(zip(idx.cpu().tolist(), conf.cpu().tolist()))

    # ---------- write out ----------
    out_path = args.output_file or os.path.join(args.model_dir, "predictions.jsonl")
    with open(out_path, "w", encoding="utf-8") as fw:
        for idx, score in preds:
            fw.write(json.dumps({"prediction": id2label[idx], "score": float(score)}) + "\n")
    print(f"Wrote {len(preds)} predictions → {out_path}")


if __name__ == "__main__":
    main()
