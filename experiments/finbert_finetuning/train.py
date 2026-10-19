"""Controlled Fine-Tuning of ProsusAI/finbert on Financial PhraseBank.

Experiment isolation: All models, logs, and checkpoints are stored in experiments/finbert_finetuning/.
No changes are made to production code or global configs.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
LABELS = ["positive", "negative", "neutral"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PhraseBankDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        text = item["text"]
        label_str = item["label"]
        label_id = LABEL2ID[label_str]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {}
    cm = {true_lbl: {pred_lbl: 0 for pred_lbl in LABELS} for true_lbl in LABELS}
    correct = 0
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1
        if t == p:
            correct += 1

    accuracy = correct / n
    per_class = {}
    f1_sum = 0.0

    for lbl in LABELS:
        tp = cm[lbl][lbl]
        fp = sum(cm[other][lbl] for other in LABELS if other != lbl)
        fn = sum(cm[lbl][other] for other in LABELS if other != lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[lbl] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": sum(cm[lbl].values())}
        f1_sum += f1

    macro_f1 = f1_sum / len(LABELS)
    return {
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def evaluate(model, dataloader, device) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * len(labels)

            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            y_pred.extend([ID2LABEL[p] for p in preds])
            y_true.extend([ID2LABEL[l] for l in labels.cpu().tolist()])

    avg_loss = total_loss / len(y_true)
    metrics = compute_metrics(y_true, y_pred)
    return avg_loss, metrics


def main():
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    splits_dir = os.path.join(exp_dir, "data", "splits")
    output_dir = os.path.join(exp_dir, "best_checkpoint")
    log_file = os.path.join(exp_dir, "training_log.json")
    os.makedirs(output_dir, exist_ok=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    model_name = "ProsusAI/finbert"
    batch_size = 16
    epochs = 3
    lr = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1

    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    train_dataset = PhraseBankDataset(os.path.join(splits_dir, "train.jsonl"), tokenizer)
    dev_dataset = PhraseBankDataset(os.path.join(splits_dir, "dev.jsonl"), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"Starting training: {len(train_dataset)} samples, {epochs} epochs, {total_steps} total steps...")

    best_dev_macro_f1 = -1.0
    training_history = []
    overall_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(labels)

            if step % 25 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{epochs} | Step {step}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_dataset)
        dev_loss, dev_metrics = evaluate(model, dev_loader, device)
        epoch_duration = time.time() - epoch_start

        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "dev_loss": round(dev_loss, 4),
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_per_class": dev_metrics["per_class"],
            "epoch_duration_sec": round(epoch_duration, 2),
        }
        training_history.append(log_entry)

        print(
            f"Epoch {epoch} complete in {epoch_duration:.1f}s | "
            f"Train Loss: {avg_train_loss:.4f} | Dev Loss: {dev_loss:.4f} | "
            f"Dev Acc: {dev_metrics['accuracy']:.4f} | Dev Macro-F1: {dev_metrics['macro_f1']:.4f}"
        )

        # Checkpoint best model
        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            print(f"--> New best Dev Macro-F1: {best_dev_macro_f1:.4f}. Saving checkpoint to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    total_duration = time.time() - overall_start_time
    print(f"\nTraining completed in {total_duration:.1f}s ({total_duration/60:.2f} mins). Best Dev Macro-F1: {best_dev_macro_f1:.4f}")

    final_log = {
        "model_name": model_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "total_duration_sec": round(total_duration, 2),
        "best_dev_macro_f1": best_dev_macro_f1,
        "history": training_history,
    }
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(final_log, f, indent=2)
    print(f"Training log saved to {log_file}")


if __name__ == "__main__":
    main()
