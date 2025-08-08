#!/usr/bin/env python
"""ft.py
Fine-tune **ModernBERT-large** on a 3-class EDU relation task with:
- Explicit tokenizer [SEP] inside a contextual prompt
- Weighted cross-entropy
- Early stopping + load-best-model-at-end
- Warmup + mixed precision
- Stable label mapping (support=0, rebuttal=1, none=2)

Usage (train):
--------------
python modernbert_finetune.py \
    --train_file train.jsonl \
    --val_file val.jsonl \
    --text1_column edu1 \
    --text2_column edu2 \
    --label_column label \
    --output_dir ./modbert_ctx_weighted \
    --freeze_encoder false \
    --wandb_project ModernBERT-FT \
    --run_name ctx_weighted

Usage (predict):
----------------
python modernbert_finetune.py \
    --mode predict \
    --model_dir ./modbert_ctx_weighted/best_model \
    --predict_file test.jsonl \
    --text1_column edu1 \
    --text2_column edu2 \
    --batch_size 64
"""
import argparse
import gc
import json
import os
import re
import math
import shutil
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    ProgressCallback,
    TrainerCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from collections import Counter


try:
    import wandb
except Exception:
    wandb = None

MODEL_NAME = "answerdotai/ModernBERT-large"

# Stable mapping to match your earlier pipeline exactly
FIXED_LABEL2ID: Dict[str, int] = {"support": 0, "rebuttal": 1, "none": 2}
ID2LABEL = {v: k for k, v in FIXED_LABEL2ID.items()}
LITERAL_SEP = "[SEP]"


class WeightedLossTrainer(Trainer):
    """HF Trainer with per-class weights for CrossEntropyLoss."""
    def __init__(self, class_weights: torch.Tensor, label_smoothing: float, rebuttal_weight_scale: float, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.rebuttal_weight_scale = rebuttal_weight_scale
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device), label_smoothing=self.label_smoothing)
        if self.rebuttal_weight_scale is not None:
            loss_fct.pos_weight = self.rebuttal_weight_scale
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class LossProgressCallback(ProgressCallback):
    """Custom progress bar callback that shows train and eval loss on the tqdm bar."""
    def __init__(self):
        super().__init__()
        self.latest_train_loss = None
        self.latest_eval_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        logs = logs or {}
        if "loss" in logs:
            try:
                self.latest_train_loss = float(logs["loss"])
            except Exception:
                pass
        if "eval_loss" in logs:
            try:
                self.latest_eval_loss = float(logs["eval_loss"])
            except Exception:
                pass

        # Build postfix string
        pieces = []
        if self.latest_train_loss is not None:
            pieces.append(f"train_loss={self.latest_train_loss:.4f}")
        if self.latest_eval_loss is not None:
            pieces.append(f"eval_loss={self.latest_eval_loss:.4f}")
        postfix = "  ".join(pieces)

        # Update training/eval bars if present
        try:
            if getattr(self, "training_bar", None) is not None:
                self.training_bar.set_postfix_str(postfix, refresh=False)
        except Exception:
            pass
        try:
            if getattr(self, "eval_bar", None) is not None:
                self.eval_bar.set_postfix_str(postfix, refresh=False)
        except Exception:
            pass

        return super().on_log(args, state, control, logs=logs, **kwargs)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ModernBERT-large for EDU relation classification")

    # I/O
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--val_file", type=str, default="val.jsonl")
    p.add_argument("--predict_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./modbert_output")
    p.add_argument("--model_dir", type=str, default=None)

    # Data columns
    p.add_argument("--text1_column", type=str, default="edu1")
    p.add_argument("--text2_column", type=str, default="edu2")
    p.add_argument("--label_column", type=str, default="label")

    # Hparams
    p.add_argument("--num_labels", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--freeze_encoder", type=str, default="false", choices=["true", "false"])
    p.add_argument("--class_weights", type=str, default=None,
                  help="Comma-separated weights, e.g. '0.2,1.0,2.3'. If omitted, computed from training data.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--rebuttal_weight_scale", type=float, default=1.0)

    p.add_argument("--encoding_mode", type=str,
                choices=["literal_sep_maxlen", "tokenizer_sep_dynamic"],
                default="literal_sep_maxlen",
                help="How to build inputs: literal [SEP] + maxlen padding vs tokenizer sep + dynamic padding")


    # Mode
    p.add_argument("--mode", choices=["train", "predict"], default="train")
    p.add_argument("--wandb_project", type=str, default="ModernBERT-FT",
                  help="If set, metrics are logged to this W&B project")
    p.add_argument("--run_name", type=str, default=None)

    args = p.parse_args()
    args.freeze_encoder = args.freeze_encoder.lower() == "true"
    return args


def get_sep_token(tokenizer) -> str:
    # Use model-native SEP; fall back sanely if missing
    return tokenizer.sep_token or tokenizer.special_tokens_map.get("sep_token") or "[SEP]"


def load_text_dataset_with_labels(
    file_path: str,
    text1_col: str,
    text2_col: str,
    label_col: str,
    tokenizer,
    encoding_mode: str,
    max_length: int = 512,
    label2id: Optional[Dict[str, int]] = None,
):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".csv", ".tsv"}:
        ds = load_dataset("csv", data_files={"data": file_path})["data"]
    elif ext in {".json", ".jsonl"}:
        ds = load_dataset("json", data_files={"data": file_path})["data"]
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Fixed mapping by default (matches earlier experiments)
    if label2id is None:
        label2id = FIXED_LABEL2ID

    # Map labels -> ints (batched)
    def lbl_fn(batch):
        return {"labels": [label2id[str(l)] for l in batch[label_col]]}
    ds = ds.map(lbl_fn, batched=True, remove_columns=[label_col])

    # Build contextual prompt with explicit SEP
    if encoding_mode == "literal_sep_maxlen":
        def build_prompt(batch):
            return {"text": [
                f"Statement 1: {e1} {LITERAL_SEP} Statement 2: {e2} {LITERAL_SEP} Relationship:"
                for e1, e2 in zip(batch[text1_col], batch[text2_col])
            ]}
        ds = ds.map(build_prompt, batched=True, remove_columns=[text1_col, text2_col])
        ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=max_length),
                    batched=True, remove_columns=["text"])  # max-length padding
    else:  # tokenizer_sep_dynamic
        sep = get_sep_token(tokenizer)
        def build_prompt(batch):
            return {"text": [
                f"Statement 1: {e1} {sep} Statement 2: {e2} {sep} Relationship:"
                for e1, e2 in zip(batch[text1_col], batch[text2_col])
            ]}
        ds = ds.map(build_prompt, batched=True, remove_columns=[text1_col, text2_col])
        ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=max_length),
                    batched=True, remove_columns=["text"])  # dynamic padding at collator
    return ds, label2id


def load_text_dataset_for_predict(
    file_path: str,
    text1_col: str,
    text2_col: str,
    tokenizer,
    max_length: int = 512,
):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".csv", ".tsv"}:
        ds = load_dataset("csv", data_files={"data": file_path})["data"]
    elif ext in {".json", ".jsonl"}:
        ds = load_dataset("json", data_files={"data": file_path})["data"]
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    sep = get_sep_token(tokenizer)
    def build_prompt(batch):
        return {"text": [
            f"Statement 1: {e1} {sep} Statement 2: {e2} {sep} Relationship:"
            for e1, e2 in zip(batch[text1_col], batch[text2_col])
        ]}
    ds = ds.map(build_prompt, batched=True, remove_columns=[text1_col, text2_col])

    ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=max_length),
                batched=True, remove_columns=["text"])
    return ds


def compute_weights(labels_column, num_classes: int):
    """Balanced weights, robust if a class is missing in the split."""
    counts = Counter(int(i) for i in labels_column)
    total = sum(counts.values())
    w = [ (total / (num_classes * counts.get(c, 0))) if counts.get(c, 0) > 0 else 1.0
          for c in range(num_classes) ]
    s = sum(w)
    w = [x * (num_classes / s) for x in w] # normalize to mean 1.0 to not overscale the loss
    return torch.tensor(w, dtype=torch.float)


def make_metrics_fn(num_labels: int):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        p_c, r_c, f1_c, _ = precision_recall_fscore_support(
            labels, preds, labels=list(range(num_labels)), zero_division=0
        )
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="micro", zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )

        metrics = {
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
        }
        # Per-class by *name* in the fixed order
        for idx, (p, r, f1) in enumerate(zip(p_c, r_c, f1_c)):
            name = ID2LABEL[idx]
            metrics[f"precision_{name}"] = p
            metrics[f"recall_{name}"] = r
            metrics[f"f1_{name}"] = f1
        return metrics
    return compute_metrics


def train(args):
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Datasets (fixed label map)
    train_ds, label2id = load_text_dataset_with_labels(
        args.train_file, args.text1_column, args.text2_column, args.label_column, tokenizer, args.encoding_mode
    )
    val_ds, _ = load_text_dataset_with_labels(
        args.val_file, args.text1_column, args.text2_column, args.label_column, tokenizer, args.encoding_mode, label2id=label2id
    )

    # Class weights (CLI overrides auto-computed)
    if args.class_weights is not None:
        weights = torch.tensor([float(w) for w in args.class_weights.split(",")], dtype=torch.float)
        assert len(weights) == args.num_labels, "Provided weights length mismatch"
    else:
        weights = torch.tensor(compute_weights(train_ds["labels"], args.num_labels), dtype=torch.float)
    print("Class weights:", weights.tolist())

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=args.num_labels,
        id2label=ID2LABEL,
        label2id=FIXED_LABEL2ID,
    )

    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    # Mixed precision selection
    amp_args = {}
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                amp_args["bf16"] = True   # Ampere+ prefers bf16
            else:
                amp_args["fp16"] = True
        except Exception:
            amp_args["fp16"] = True

    data_collator = DataCollatorWithPadding(tokenizer)

    # W&B wiring: enable if library is available and not explicitly disabled
    wandb_disabled = os.getenv("WANDB_DISABLED", "false").lower() == "true"
    use_wandb = bool(wandb) and not wandb_disabled
    # Respect CLI project if provided by exporting env var for the HF WandbCallback
    if use_wandb and args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Auto-generate a descriptive run name if none provided
    def _auto_run_name() -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        freeze_flag = "T" if args.freeze_encoder else "F"
        out_name = Path(args.output_dir).name or "modbert_output"
        return f"{out_name}_bs{args.batch_size}_lr{args.lr:g}_frz{freeze_flag}_seed{args.seed}_{timestamp}"

    run_name_to_use = args.run_name or _auto_run_name()

    steps_per_epoch = max(1, math.ceil(len(train_ds) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.09 * total_steps))  # ~9%

    eval_steps = max(1, steps_per_epoch // 2) # twice per epoch
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=6,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=eval_steps,
        logging_first_step=True,
        report_to=["wandb"] if use_wandb else [],
        run_name=run_name_to_use,
        **amp_args,
    )

    trainer = WeightedLossTrainer(
        class_weights=weights,
        label_smoothing=args.label_smoothing,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_metrics_fn(args.num_labels),
        rebuttal_weight_scale=args.rebuttal_weight_scale,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # Replace the default progress callback with our loss-postfix variant
    try:
        trainer.remove_callback(ProgressCallback)
    except Exception:
        # Fallback: manually filter it out if API differs
        try:
            trainer.callback_handler.callbacks = [
                cb for cb in trainer.callback_handler.callbacks if not isinstance(cb, ProgressCallback)
            ]
        except Exception:
            pass
    trainer.add_callback(LossProgressCallback())

    # Let HF's WandbCallback manage the W&B run; no manual wandb.init to avoid conflicts

    trainer.train()

    # Evaluate best checkpoint on val
    best_metrics = trainer.evaluate(eval_dataset=val_ds)
    best_ckpt = trainer.state.best_model_checkpoint

    # Estimate best epoch from checkpoint step
    best_epoch = None
    if best_ckpt:
        m = re.search(r"checkpoint-(\d+)", best_ckpt)
        if m:
            best_step = int(m.group(1))
            steps_per_epoch = max(1, math.ceil(len(train_ds) / args.batch_size))
            best_epoch = best_step / steps_per_epoch

    if best_epoch is not None:
        print(f"\n🏅  Best epoch ≈ {best_epoch:.1f}")
    else:
        print("\n🏅  Best epoch: unavailable (could not parse from checkpoint)")

    print("🔍  Best-checkpoint metrics:")
    for k, v in best_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Copy best checkpoint to a stable path
    if best_ckpt:
        print(f"\n🏆  Best checkpoint: {best_ckpt}")
        best_dir = os.path.join(args.output_dir, "best_model")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(best_ckpt, best_dir)

        # Save tokenizer + label map alongside
        tokenizer.save_pretrained(best_dir)
        with open(os.path.join(best_dir, "label2id.json"), "w") as f:
            json.dump(FIXED_LABEL2ID, f, indent=2)

        with open(os.path.join(args.output_dir, "BEST_CHECKPOINT.txt"), "w") as f:
            f.write(best_dir + "\n")
    else:
        best_dir = args.output_dir

    # Save final model (now equal to best because load_best_model_at_end=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(FIXED_LABEL2ID, f, indent=2)

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

    # Try to load saved label2id; fall back to fixed map
    label2id_path = os.path.join(args.model_dir, "label2id.json")
    if os.path.exists(label2id_path):
        with open(label2id_path) as f:
            label2id = json.load(f)
        id2label = {int(v): k for k, v in label2id.items()}
    else:
        id2label = ID2LABEL

    # Load prediction texts (no labels required)
    pred_ds = load_text_dataset_for_predict(
        args.predict_file, args.text1_column, args.text2_column, tokenizer
    )
    collator = DataCollatorWithPadding(tokenizer)
    loader = torch.utils.data.DataLoader(pred_ds, batch_size=args.batch_size, collate_fn=collator)

    preds_all = []
    with torch.no_grad():
        for batch in loader:
            # No labels are passed
            inputs = {k: v.to(device) for k, v in batch.items()}
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            preds_all.extend(preds)

    out_path = os.path.join(args.model_dir, "predictions.jsonl")
    with open(out_path, "w") as fw:
        for pred in preds_all:
            fw.write(json.dumps({"prediction": id2label[int(pred)]}) + "\n")
    print(f"Saved predictions for {len(preds_all)} texts to {out_path}")


def clear_gpu_memory() -> None:
    """Best-effort CUDA memory cleanup for fresh runs in the same process."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


if __name__ == "__main__":
    clear_gpu_memory()
    _args = parse_args()
    if _args.mode == "train":
        train(_args)
    else:
        predict(_args)
