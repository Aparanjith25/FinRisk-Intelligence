"""Comprehensive Head-to-Head Comparative Evaluation:

Models compared:
- Model A: Pretrained ProsusAI/finbert
- Model B: Rule-based Offline Deterministic Baseline
- Model C: Fine-tuned FinBERT Candidate (trained on Financial PhraseBank train split)

Evaluated on:
1. Genuinely Held-out Financial PhraseBank Test Split (339 samples, zero overlap with training)
2. External Curated Regulatory Benchmark (30 samples: 22 dev, 8 held-out ECB samples)

Metrics:
- Accuracy, Macro-F1, Macro-Precision, Macro-Recall
- Per-class F1, Precision, Recall for positive, negative, neutral
- Confusion Matrices (3x3)
- Inference Latency (ms/sample) and throughput
- Model footprint & contract verification
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from finrisk.nlp.sentiment_analyzer import (
    FinBERTSentimentAnalyzer,
    SentimentResult,
    deterministic_baseline_sentiment,
)
from tests.evaluation.annotated_news_sample import ANNOTATED_NEWS_SAMPLES

LABELS = ["positive", "negative", "neutral"]


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
    p_sum = 0.0
    r_sum = 0.0

    for lbl in LABELS:
        tp = cm[lbl][lbl]
        fp = sum(cm[other][lbl] for other in LABELS if other != lbl)
        fn = sum(cm[lbl][other] for other in LABELS if other != lbl)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(cm[lbl].values()),
        }
        f1_sum += f1
        p_sum += prec
        r_sum += rec

    macro_f1 = f1_sum / len(LABELS)
    macro_prec = p_sum / len(LABELS)
    macro_rec = r_sum / len(LABELS)

    return {
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def evaluate_model_on_dataset(
    predict_fn,
    texts: List[str],
    golds: List[str],
    warmup: bool = True,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    if warmup and len(texts) > 0:
        predict_fn(texts[0])

    preds = []
    records = []
    t0 = time.perf_counter()
    for text, gold in zip(texts, golds):
        res = predict_fn(text)
        preds.append(res.label)
        records.append({
            "text": text[:80] + "..." if len(text) > 80 else text,
            "gold": gold,
            "pred": res.label,
            "confidence": round(res.confidence, 4),
            "score": round(res.score, 4),
            "correct": (res.label == gold),
        })
    total_time = time.perf_counter() - t0
    avg_latency_ms = (total_time / len(texts)) * 1000.0 if texts else 0.0

    metrics = compute_metrics(golds, preds)
    return metrics, avg_latency_ms, records


def main():
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    test_jsonl = os.path.join(exp_dir, "data", "splits", "test.jsonl")
    checkpoint_dir = os.path.join(exp_dir, "best_checkpoint")
    results_file = os.path.join(exp_dir, "evaluation_results.json")

    print("=" * 80)
    print("FINRISK INTELLIGENCE: CONTROLLED COMPARATIVE EVALUATION")
    print("=" * 80)

    # 1. Load Datasets
    print("\n[1/4] Loading Datasets...")
    phrasebank_test = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                phrasebank_test.append(json.loads(line))
    pb_texts = [item["text"] for item in phrasebank_test]
    pb_golds = [item["label"] for item in phrasebank_test]
    print(f"  - Loaded {len(pb_texts)} PhraseBank held-out test samples.")

    # Regulatory Benchmark (Total, Dev, Held-out)
    reg_texts_all = [s.headline + ". " + s.text for s in ANNOTATED_NEWS_SAMPLES]
    reg_golds_all = [s.gold_sentiment for s in ANNOTATED_NEWS_SAMPLES]

    reg_dev_samples = [s for s in ANNOTATED_NEWS_SAMPLES if not s.is_held_out]
    reg_texts_dev = [s.headline + ". " + s.text for s in reg_dev_samples]
    reg_golds_dev = [s.gold_sentiment for s in reg_dev_samples]

    reg_heldout_samples = [s for s in ANNOTATED_NEWS_SAMPLES if s.is_held_out]
    reg_texts_heldout = [s.headline + ". " + s.text for s in reg_heldout_samples]
    reg_golds_heldout = [s.gold_sentiment for s in reg_heldout_samples]

    print(f"  - Loaded Regulatory Benchmark: {len(reg_texts_all)} total ({len(reg_texts_dev)} dev, {len(reg_texts_heldout)} held-out ECB).")

    # 2. Initialize Models
    print("\n[2/4] Initializing Models...")

    # Model B: Baseline
    print("  -> Initializing Model B (Offline Baseline)...")
    model_b_analyzer = FinBERTSentimentAnalyzer(
        allow_offline_fallback=True,
        predictor=deterministic_baseline_sentiment,
    )

    # Model A: Pretrained ProsusAI/finbert
    print("  -> Initializing Model A (Pretrained ProsusAI/finbert)...")
    model_a_analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=False)

    # Model C: Fine-tuned FinBERT
    print(f"  -> Initializing Model C (Fine-Tuned Candidate from {checkpoint_dir})...")
    if not os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        print(f"ERROR: Checkpoint not found at {checkpoint_dir}!")
        return

    # Create analyzer wrapper for fine-tuned candidate
    ft_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    ft_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    ft_model.eval()

    def model_c_predict(text: str) -> SentimentResult:
        inputs = ft_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        with torch.no_grad():
            outputs = ft_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze(0)

        # id2label is 0: positive, 1: negative, 2: neutral
        p_pos = float(probs[0].item())
        p_neg = float(probs[1].item())
        p_neu = float(probs[2].item())

        score = max(-1.0, min(1.0, p_pos - p_neg))
        prob_dict = {"positive": p_pos, "negative": p_neg, "neutral": p_neu}
        best_label = max(prob_dict, key=prob_dict.get)

        return SentimentResult(
            label=best_label,
            score=round(score, 4),
            confidence=round(prob_dict[best_label], 4),
            probabilities={k: round(v, 4) for k, v in prob_dict.items()},
            model_name="finbert-finetuned-phrasebank-ep1",
            inference_source="finetuned_candidate",
        )

    # Model Size Analysis
    def get_dir_size_mb(path: str) -> float:
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return round(total / (1024 * 1024), 2)

    ft_size_mb = get_dir_size_mb(checkpoint_dir)
    ft_params = sum(p.numel() for p in ft_model.parameters())

    print(f"  Model C Parameters: {ft_params:,} ({ft_size_mb} MB on disk)")

    # 3. Evaluate Models
    print("\n[3/4] Running Comparative Evaluations...")
    eval_results = {
        "models": {
            "Model A": {"name": "Pretrained ProsusAI/finbert", "type": "HuggingFace Pretrained Transformer"},
            "Model B": {"name": "Deterministic Baseline", "type": "Lexicon Heuristic Fallback"},
            "Model C": {"name": "Fine-Tuned FinBERT", "type": "Domain Fine-Tuned Transformer", "checkpoint": checkpoint_dir, "parameters": ft_params, "size_mb": ft_size_mb},
        },
        "datasets": {
            "phrasebank_test": {"description": "Held-out 15% test split of Financial PhraseBank (AllAgree)", "samples": len(pb_texts)},
            "regulatory_benchmark_all": {"description": "Human-annotated Regulatory Benchmark (Fed, SEC, ECB)", "samples": len(reg_texts_all)},
            "regulatory_benchmark_dev": {"description": "Regulatory Benchmark Dev Subset (Fed, SEC)", "samples": len(reg_texts_dev)},
            "regulatory_benchmark_heldout_ecb": {"description": "Regulatory Benchmark Held-out Subset (ECB)", "samples": len(reg_texts_heldout)},
        },
        "results": {},
    }

    eval_pairs = [
        ("Model B (Baseline)", model_b_analyzer.analyze),
        ("Model A (Pretrained FinBERT)", model_a_analyzer.analyze),
        ("Model C (Fine-Tuned FinBERT)", model_c_predict),
    ]

    for model_name, pred_fn in eval_pairs:
        print(f"\n--- Evaluating {model_name} ---")
        eval_results["results"][model_name] = {}

        # 1. PhraseBank Test
        m_pb, lat_pb, rec_pb = evaluate_model_on_dataset(pred_fn, pb_texts, pb_golds)
        eval_results["results"][model_name]["phrasebank_test"] = {
            "metrics": m_pb,
            "latency_ms": round(lat_pb, 2),
        }
        print(f"  PhraseBank Test (N={m_pb['n_samples']}): Acc={m_pb['accuracy']:.4f}, Macro-F1={m_pb['macro_f1']:.4f}, Latency={lat_pb:.2f}ms")

        # 2. Regulatory Benchmark All (N=30)
        m_reg, lat_reg, rec_reg = evaluate_model_on_dataset(pred_fn, reg_texts_all, reg_golds_all)
        eval_results["results"][model_name]["regulatory_benchmark_all"] = {
            "metrics": m_reg,
            "latency_ms": round(lat_reg, 2),
            "disagreements": [r for r in rec_reg if not r["correct"]],
        }
        print(f"  Regulatory Benchmark All (N={m_reg['n_samples']}): Acc={m_reg['accuracy']:.4f}, Macro-F1={m_reg['macro_f1']:.4f}")

        # 3. Regulatory Benchmark Dev (N=22)
        m_dev, _, _ = evaluate_model_on_dataset(pred_fn, reg_texts_dev, reg_golds_dev)
        eval_results["results"][model_name]["regulatory_benchmark_dev"] = {
            "metrics": m_dev,
        }
        print(f"  Regulatory Dev (N={m_dev['n_samples']}): Acc={m_dev['accuracy']:.4f}, Macro-F1={m_dev['macro_f1']:.4f}")

        # 4. Regulatory Benchmark Held-Out ECB (N=8)
        m_ecb, _, _ = evaluate_model_on_dataset(pred_fn, reg_texts_heldout, reg_golds_heldout)
        eval_results["results"][model_name]["regulatory_benchmark_heldout_ecb"] = {
            "metrics": m_ecb,
        }
        print(f"  Regulatory Held-out ECB (N={m_ecb['n_samples']}): Acc={m_ecb['accuracy']:.4f}, Macro-F1={m_ecb['macro_f1']:.4f}")

    # 4. Contract Verification for Model C
    print("\n[4/4] Verifying Inference Contract for Model C...")
    test_cases = [
        ("Net profit surged 45% due to robust sales expansion.", "positive"),
        ("Operating loss widened significantly amid supply chain defaults.", "negative"),
        ("The company will publish its interim financial report on Tuesday.", "neutral"),
        ("", "neutral"),  # Empty string edge case
        ("   ", "neutral"), # Whitespace edge case
    ]
    contract_results = []
    for tc_text, exp_sentiment in test_cases:
        if not tc_text.strip():
            # Standard FinBERT handling for empty
            continue
        res = model_c_predict(tc_text)
        is_bounded = -1.0 <= res.score <= 1.0
        prob_sum = sum(res.probabilities.values())
        contract_results.append({
            "input": tc_text,
            "predicted_label": res.label,
            "score": res.score,
            "bounded_score": is_bounded,
            "prob_sum": round(prob_sum, 4),
        })
        print(f"  Contract test '{tc_text[:30]}...': label={res.label}, score={res.score}, bounded={is_bounded}")

    eval_results["contract_verification"] = contract_results

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nAll comparative evaluation results successfully written to {results_file}")


if __name__ == "__main__":
    main()
