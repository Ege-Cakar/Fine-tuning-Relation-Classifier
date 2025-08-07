#!/usr/bin/env python
"""modernbert_finetune.py
A unified script for fine‑tuning **ModernBERT‑large** on a 3‑class text‑classification task _and_ running batch inference afterwards.

Usage examples:
---------------
Train by **freezing** the encoder and only training the classifier:
```
python modernbert_finetune.py \
    --train_file train.csv \
    --val_file val.csv \
    --text_column text \
    --label_column label \
    --output_dir ./modbert_cls_freeze \
    --freeze_encoder true
```

Train by fine‑tuning the **entire** model + classifier:
```
python modernbert_finetune.py \
    --train_file train.csv \
    --val_file val.csv \
    --text_column text \
    --label_column label \
    --output_dir ./modbert_cls_full \
    --freeze_encoder false
```

Run batch inference after training (works for either checkpoint):
```
python modernbert_finetune.py \
    --mode predict \
    --model_dir ./modbert_cls_freeze \
    --predict_file test.jsonl \
    --text_column text \
    --batch_size 64
```

Requirements
------------
```bash
pip install "transformers>=4.42" datasets accelerate torch scikit-learn tqdm
```

The script automatically detects GPU/CPU and supports multi‑GPU and mixed‑precision via `accelerate`.
"""
import argparse
import json
import os
from collections import Counter
from typing import List

import torch
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import shutil
import wandb

MODEL_NAME = "answerdotai/ModernBERT-large"  # default base checkpoint


class WeightedLossTrainer(Trainer):
    """Hugging Face Trainer with per‑class weights for CrossEntropyLoss."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine‑tune ModernBERT‑large for multi‑class classification")

    # I/O
    parser.add_argument("--train_file", type=str, default="train.jsonl", help="Path to training CSV/JSONL file")
    parser.add_argument("--val_file", type=str, default="val.jsonl", help="Path to validation CSV/JSONL file")
    parser.add_argument("--predict_file", type=str, default=None, help="Path to file with texts for prediction (JSONL/CSV)")
    parser.add_argument("--output_dir", type=str, default="./modbert_output", help="Where to save checkpoints")
    parser.add_argument("--model_dir", type=str, default=None, help="Directory of a fine‑tuned model to load for prediction")

    # Data columns
    parser.add_argument("--text1_column", type=str, default="edu1",
                    help="First text column (e.g. edu1)")
    parser.add_argument("--text2_column", type=str, default="edu2",
                        help="Second text column (e.g. edu2)")
    parser.add_argument("--label_column", type=str, default="label")

    # Hyper‑parameters
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--freeze_encoder", type=str, default="false", choices=["true", "false"], help="Freeze backbone? only train classifier")
    parser.add_argument("--class_weights", type=str, default=None, help="Comma‑separated weights, e.g. '0.2,1.0,2.3'. If omitted, computed from training data.")
    parser.add_argument("--seed", type=int, default=42)

    # Mode
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--wandb_project", type=str, default="ModernBERT-FT",
                    help="If set, metrics are logged to this W&B project")
    parser.add_argument("--run_name", type=str, default="Run-1",
                    help="Optional wandb run name (defaults to output_dir)")

    args = parser.parse_args()
    args.freeze_encoder = args.freeze_encoder.lower() == "true"
    return args

def build_prompt(batch, col1, col2):
    # batch is a *list* when batched=True below
    return {"text": [
        f"Argument 1: {e1}, Argument 2: {e2}, Relationship:"
        for e1, e2 in zip(batch[col1], batch[col2])
    ]}

def load_text_dataset(file_path: str,
                      text1_col: str,
                      text2_col: str,
                      label_col: str,
                      label2id: dict | None = None):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".csv", ".tsv"}:
        ds = load_dataset("csv", data_files={"data": file_path})["data"]
    elif ext in {".json", ".jsonl"}:
        ds = load_dataset("json", data_files={"data": file_path})["data"]
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # ---- map labels -> ints (batched!)
    if label2id is None:
        unique = sorted({str(l) for l in ds[label_col]})
        label2id = {lbl: i for i, lbl in enumerate(unique)}

    def lbl_fn(batch):
        return {"labels": [label2id[str(l)] for l in batch[label_col]]}

    ds = ds.map(lbl_fn, batched=True, remove_columns=[label_col])

    # ---- build the combined prompt and drop raw EDU columns
    ds = ds.map(lambda b: build_prompt(b, text1_col, text2_col),
                batched=True,
                remove_columns=[text1_col, text2_col])

    print(ds.features) 

    return ds, label2id



def tokenize_function(examples, tokenizer, text_column):
    return tokenizer(examples[text_column], truncation=True, max_length=512)


def compute_weights(labels_column, num_classes: int):
    """
    Convert the Arrow column to a NumPy array and feed sklearn
    (its API now insists on ndarray for `classes`).
    """
    y = np.asarray(labels_column, dtype=int)
    classes = np.arange(num_classes)
    return compute_class_weight(class_weight="balanced",
                                classes=classes,
                                y=y)

def train(args):
    set_seed(args.seed)

    # Load dataset
    train_ds, label2id = load_text_dataset(args.train_file,
                                       args.text1_column,
                                       args.text2_column,
                                       args.label_column)
    val_ds, _ = load_text_dataset(args.val_file,
                              args.text1_column,
                              args.text2_column,
                              args.label_column,
                              label2id)

    id2label = {v: k for k, v in label2id.items()}

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # per-class
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(
            labels, preds, labels=list(range(args.num_labels)),
            zero_division=0
        )

        # overall
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="micro", zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )

        # build metrics dict
        metrics = {
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
        }
        for idx, (p, r, f1) in enumerate(zip(p_c, r_c, f1_c)):
            name = id2label[idx]
            metrics[f"precision_{name}"] = p
            metrics[f"recall_{name}"] = r
            metrics[f"f1_{name}"] = f1
        return metrics

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer, "text"), batched=True, remove_columns=["text"])
    val_ds   = val_ds.map(lambda x: tokenize_function(x, tokenizer, "text"), batched=True, remove_columns=["text"])
    # Prepare class weights
    if args.class_weights is not None:
        weights = torch.tensor([float(w) for w in args.class_weights.split(",")])
        assert len(weights) == args.num_labels, "Provided weights length mismatch"
    else:
        weights = torch.tensor(compute_weights(train_ds["labels"], args.num_labels), dtype=torch.float)
    print("Class weights:", weights)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=args.num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    # Data collator automatically pads batches
    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=["wandb"] if args.wandb_project else [],
        run_name=(args.run_name or os.path.basename(args.output_dir))
                 if args.wandb_project else None,

    )

    trainer = WeightedLossTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    best_metrics = trainer.evaluate(eval_dataset=val_ds)
    best_epoch   = trainer.state.best_step / len(train_ds) * args.batch_size
    print(f"\n🏅  Best epoch ≈ {best_epoch:.1f}")
    print("🔍  Best-checkpoint metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")


    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:                          # should always be non-None because
        print(f"\n🏆  Best checkpoint: {best_ckpt}")  # we set load_best_model_at_end=True
        best_dir = os.path.join(args.output_dir, "best_model")
    
        # Make a fresh copy of the entire checkpoint folder
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(best_ckpt, best_dir)
    
        # Copy tokenizer + label map alongside
        tokenizer.save_pretrained(best_dir)
        with open(os.path.join(best_dir, "label2id.json"), "w") as f:
            json.dump(label2id, f, indent=2)
    
        # Convenience pointer
        with open(os.path.join(args.output_dir, "BEST_CHECKPOINT.txt"), "w") as f:
            f.write(best_dir + "\n")
    else:
        best_dir = args.output_dir  # fallback – shouldn't happen
    
    # Save the final model (includes tokenizer + config)
    trainer.save_model(args.output_dir)     # root dir now equals best model
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)
    
    print("\nTraining complete.")
    print(f" • Best model copied to  {best_dir}")
    print(f" • Root directory now holds identical weights.")


def predict(args):
    assert args.model_dir, "--model_dir is required for prediction mode"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model, tokenizer, mapping
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    with open(os.path.join(args.model_dir, "label2id.json")) as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    # Load prediction texts
    pred_ds, _ = load_text_dataset(args.predict_file,
                               args.text1_column,
                               args.text2_column,
                               args.label_column)

    pred_ds = pred_ds.map(lambda x: tokenize_function(x, tokenizer, "text"),
                          batched=True, remove_columns=["text"])
    pred_dataloader = torch.utils.data.DataLoader(pred_ds, batch_size=args.batch_size, collate_fn=DataCollatorWithPadding(tokenizer))

    all_preds = []
    with torch.no_grad():
        for batch in pred_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)

    # Write to file
    out_path = os.path.join(args.model_dir, "predictions.jsonl")
    with open(out_path, "w") as fw:
        for pred in all_preds:
            fw.write(json.dumps({"prediction": id2label[pred]}) + "\n")
    print(f"Saved predictions for {len(all_preds)} texts to {out_path}")


if __name__ == "__main__":
    _args = parse_args()
    if _args.wandb_project is not None:
        wandb.init(project=_args.wandb_project,
                   name=_args.run_name or os.path.basename(_args.output_dir),
                   config=vars(_args))

    if _args.mode == "train":
        train(_args)
    else:
        predict(_args)
