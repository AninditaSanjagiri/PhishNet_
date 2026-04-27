"""
training/train_text_model.py
==============================
Fine-tune DistilBERT on public phishing text datasets.

Public datasets used (all free, no login):
  1. HuggingFace: "ealvaradob/phishing-dataset-multilingual"
     → already labelled phishing/legit email + web text
  2. HuggingFace: "zefang-liu/phishing-email-dataset"
     → English phishing emails with binary labels
  Both via: from datasets import load_dataset

Output:
  models/distilbert_phishing/     ← HuggingFace checkpoint dir
  evaluation/text_bert_metrics.json

Usage:
  python training/train_text_model.py --epochs 3 --batch-size 16
  python training/train_text_model.py --fast          # 1 epoch, small batch, quick test
"""
import argparse, json, time, sys
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"
EVAL_DIR   = Path(__file__).parent.parent / "evaluation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--fast",       action="store_true",
                        help="1 epoch, 500 samples — quick smoke test")
    args = parser.parse_args()

    if args.fast:
        args.epochs     = 1
        args.batch_size = 8

    # ── Imports (lazy to fail fast if not installed) ──────────────────────
    try:
        import torch
        from datasets import load_dataset, concatenate_datasets, Dataset
        from transformers import (AutoTokenizer,
                                   AutoModelForSequenceClassification,
                                   TrainingArguments, Trainer,
                                   DataCollatorWithPadding)
        from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                      roc_auc_score)
    except ImportError as exc:
        print(f"❌ Missing dependency: {exc}")
        print("   pip install transformers datasets accelerate torch scikit-learn")
        sys.exit(1)

    MODEL_NAME = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Load datasets ─────────────────────────────────────────────────────
    print("Loading phishing text datasets from HuggingFace…")
    parts = []

    # Dataset 1: phishing email dataset
    try:
        ds1 = load_dataset("zefang-liu/phishing-email-dataset", split="train")
        # Columns: text, label (1=phishing, 0=legit)
        if "text" in ds1.column_names and "label" in ds1.column_names:
            parts.append(ds1)
            print(f"  zefang-liu: {len(ds1)} samples")
    except Exception as exc:
        print(f"  ⚠️  zefang-liu failed: {exc}")

    # Dataset 2: multilingual phishing (English subset)
    try:
        ds2 = load_dataset("ealvaradob/phishing-dataset-multilingual",
                           "english_dataset", split="train")
        # Map to standard text/label columns
        if "text" not in ds2.column_names:
            # Try common column name variants
            for col in ["url","email","content","page_text"]:
                if col in ds2.column_names:
                    ds2 = ds2.rename_column(col, "text")
                    break
        if "label" not in ds2.column_names:
            for col in ["labels","phishing","class","target"]:
                if col in ds2.column_names:
                    ds2 = ds2.rename_column(col, "label")
                    break
        if "text" in ds2.column_names and "label" in ds2.column_names:
            ds2 = ds2.select_columns(["text","label"])
            parts.append(ds2)
            print(f"  ealvaradob: {len(ds2)} samples")
    except Exception as exc:
        print(f"  ⚠️  ealvaradob failed: {exc}")

    if not parts:
        print("⚠️  No HuggingFace datasets loaded. Creating minimal synthetic set…")
        phish_texts = [
            "Your account has been suspended. Click here to verify your identity immediately.",
            "Urgent: Confirm your banking credentials to avoid account closure.",
            "You have won a prize! Enter your social security number to claim it.",
            "Your PayPal account will be closed unless you update your password now.",
            "Dear customer, your login credentials need to be verified. Click the link below.",
        ] * 100
        legit_texts = [
            "Thank you for your order. Your package will arrive in 3-5 business days.",
            "Your monthly statement is now available. Log in to view your account.",
            "Meeting scheduled for tomorrow at 2pm. Please confirm your attendance.",
            "New features have been added to your account. See what's new.",
            "Your subscription has been renewed successfully. Thank you.",
        ] * 100
        texts  = phish_texts + legit_texts
        labels = [1]*len(phish_texts) + [0]*len(legit_texts)
        raw_ds = Dataset.from_dict({"text": texts, "label": labels})
        parts  = [raw_ds]
        print(f"  Synthetic: {len(raw_ds)} samples")

    # Combine and shuffle
    from datasets import concatenate_datasets
    combined = concatenate_datasets(parts).shuffle(seed=42)

    # Fast mode: subsample
    if args.fast:
        combined = combined.select(range(min(500, len(combined))))

    print(f"\nTotal samples: {len(combined)}")

    # ── Tokenise ──────────────────────────────────────────────────────────
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True,
                         max_length=args.max_length, padding=False)

    tokenised = combined.map(tokenize, batched=True,
                              remove_columns=[c for c in combined.column_names
                                              if c not in ("label",)])
    tokenised = tokenised.rename_column("label","labels")
    tokenised.set_format("torch")

    # Train/val split
    split    = tokenised.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"  Train: {len(train_ds)}  Val: {len(eval_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds  = np.argmax(logits, axis=-1)
        probs  = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        p, r, f, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(labels, probs[:, 1])
        except Exception:
            auc = 0.0
        return {
            "accuracy":  accuracy_score(labels, preds),
            "precision": float(p), "recall": float(r),
            "f1": float(f), "roc_auc": float(auc),
        }

    # ── Training ──────────────────────────────────────────────────────────
    out_dir = str(MODELS_DIR / "distilbert_phishing")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir              = out_dir,
        num_train_epochs        = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate           = args.lr,
        weight_decay            = 0.01,
        warmup_ratio            = 0.1,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "f1",
        logging_steps           = 50,
        report_to               = "none",
        fp16                    = False,   # safe for CPU
        dataloader_num_workers  = 0,
    )

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        tokenizer        = tokenizer,
        data_collator    = DataCollatorWithPadding(tokenizer),
        compute_metrics  = compute_metrics,
    )

    print("\nFine-tuning DistilBERT…")
    t0 = time.time()
    trainer.train()
    train_time = round(time.time() - t0, 1)

    # ── Final eval ────────────────────────────────────────────────────────
    eval_result = trainer.evaluate()
    print(f"\nFinal val metrics: {eval_result}")
    print(f"Training time: {train_time}s")

    # ── Save model + tokenizer ────────────────────────────────────────────
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"✅ Model saved → {out_dir}")

    # ── Save metrics ──────────────────────────────────────────────────────
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model":       MODEL_NAME,
        "dataset_size": len(combined),
        "train_size":   len(train_ds),
        "val_size":     len(eval_ds),
        "epochs":       args.epochs,
        "train_time_s": train_time,
        **{k.replace("eval_",""):v for k,v in eval_result.items()
           if "eval_" in k},
    }
    with open(EVAL_DIR / "text_bert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved → evaluation/text_bert_metrics.json")

    # ── Update text_agent to use fine-tuned checkpoint ────────────────────
    print(f"\n📝 To use this fine-tuned model, set MODEL_NAME in text_agent.py to:")
    print(f"   '{out_dir}'")


if __name__ == "__main__":
    main()
