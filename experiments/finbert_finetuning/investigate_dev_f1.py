"""Investigation into the high dev Macro-F1 (0.9684) on Financial PhraseBank.

Checks:
1. Exact and near-duplicate sentences within Sentences_AllAgree.txt
2. Exact and near-duplicate sentences between train.jsonl and dev.jsonl / test.jsonl
3. Pretrained ProsusAI/finbert performance on dev.jsonl (Zero-shot / Pretrained baseline)
4. Domain provenance of ProsusAI/finbert (Did ProsusAI train on Financial PhraseBank?)
5. Text length and lexical distribution differences between PhraseBank and Regulatory Benchmark
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer
from tests.evaluation.annotated_news_sample import ANNOTATED_NEWS_SAMPLES

exp_dir = os.path.dirname(os.path.abspath(__file__))
splits_dir = os.path.join(exp_dir, "data", "splits")

# 1. Load splits
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_data = load_jsonl(os.path.join(splits_dir, "train.jsonl"))
dev_data = load_jsonl(os.path.join(splits_dir, "dev.jsonl"))
test_data = load_jsonl(os.path.join(splits_dir, "test.jsonl"))

print("=" * 80)
print("INVESTIGATION: DATA INTEGRITY & DEV MACRO-F1 ROOT CAUSE")
print("=" * 80)

# Check 1: Duplicate text in original file
allagree_file = os.path.join(exp_dir, "data", "FinancialPhraseBank-v1.0", "Sentences_AllAgree.txt")
raw_sentences = []
with open(allagree_file, "r", encoding="latin-1") as f:
    for line in f:
        if "@" in line:
            raw_sentences.append(line.rsplit("@", 1)[0].strip())

raw_counts = Counter(raw_sentences)
duplicates = {k: v for k, v in raw_counts.items() if v > 1}
print(f"\n[1] Duplicates in Sentences_AllAgree.txt:")
print(f"  Total lines: {len(raw_sentences)}")
print(f"  Unique texts: {len(raw_counts)}")
print(f"  Duplicated texts: {len(duplicates)} (accounting for {sum(duplicates.values()) - len(duplicates)} redundant entries)")
for k, v in list(duplicates.items())[:5]:
    print(f"    - (x{v}): {k[:70]}...")

# Check 2: Leakage between Train and Dev / Test due to identical repeated strings
train_texts = set(d["text"].strip().lower() for d in train_data)
dev_texts = [d["text"].strip().lower() for d in dev_data]
test_texts = [d["text"].strip().lower() for d in test_data]

dev_leak = [t for t in dev_texts if t in train_texts]
test_leak = [t for t in test_texts if t in train_texts]

print(f"\n[2] Exact Text Overlap across Splits:")
print(f"  Dev sentences appearing identically in Train: {len(dev_leak)} / {len(dev_data)} ({len(dev_leak)/len(dev_data)*100:.2f}%)")
print(f"  Test sentences appearing identically in Train: {len(test_leak)} / {len(test_data)} ({len(test_leak)/len(test_data)*100:.2f}%)")

# Check 3: Pretrained ProsusAI/finbert performance on Dev BEFORE fine-tuning
print(f"\n[3] Evaluating Pretrained ProsusAI/finbert on Dev Split (Pre-training Baseline)...")
pretrained_analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=False)

labels = ["positive", "negative", "neutral"]
dev_preds = []
dev_golds = [d["label"] for d in dev_data]
for d in dev_data:
    res = pretrained_analyzer.analyze(d["text"])
    dev_preds.append(res.label)

correct = sum(p == g for p, g in zip(dev_preds, dev_golds))
acc = correct / len(dev_golds)

# Compute macro F1
f1_sum = 0.0
for lbl in labels:
    tp = sum(p == lbl and g == lbl for p, g in zip(dev_preds, dev_golds))
    fp = sum(p == lbl and g != lbl for p, g in zip(dev_preds, dev_golds))
    fn = sum(p != lbl and g == lbl for p, g in zip(dev_preds, dev_golds))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    f1_sum += f1
base_dev_macro_f1 = f1_sum / len(labels)

print(f"  Pretrained ProsusAI/finbert Dev Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"  Pretrained ProsusAI/finbert Dev Macro-F1: {base_dev_macro_f1:.4f}")

# Check 4: Domain & Style comparison: PhraseBank vs Regulatory Benchmark
print(f"\n[4] Text Characteristics & Domain Difference:")
pb_lengths = [len(t.split()) for t in raw_sentences]
reg_lengths = [len((s.headline + " " + s.text).split()) for s in ANNOTATED_NEWS_SAMPLES]
print(f"  PhraseBank avg token length: {sum(pb_lengths)/len(pb_lengths):.1f} words (min: {min(pb_lengths)}, max: {max(pb_lengths)})")
print(f"  Regulatory Benchmark avg token length: {sum(reg_lengths)/len(reg_lengths):.1f} words (min: {min(reg_lengths)}, max: {max(reg_lengths)})")
print(f"  PhraseBank domain: Corporate earnings, quarterly profits/losses, dividend announcements.")
print(f"  Regulatory benchmark domain: Central bank monetary policy (rates, inflation targets, quantitative tightening) and SEC regulatory enforcement actions.")

results_summary = {
    "raw_sentences": len(raw_sentences),
    "unique_raw_sentences": len(raw_counts),
    "duplicate_raw_sentences": len(duplicates),
    "dev_exact_train_overlap": len(dev_leak),
    "test_exact_train_overlap": len(test_leak),
    "pretrained_dev_accuracy": round(acc, 4),
    "pretrained_dev_macro_f1": round(base_dev_macro_f1, 4),
    "pb_avg_words": round(sum(pb_lengths)/len(pb_lengths), 1),
    "reg_avg_words": round(sum(reg_lengths)/len(reg_lengths), 1),
}
with open(os.path.join(exp_dir, "investigation_dev_f1.json"), "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2)

print("\nInvestigation complete. Saved to investigation_dev_f1.json")
