"""Benchmark SourceAwareSentimentRouter against original Pretrained FinBERT.

Evaluates:
1. Financial PhraseBank Held-Out Test Split (N=339, source="financial_phrasebank")
2. Regulatory Risk Benchmark (N=30, sources from Fed, SEC, ECB)
3. Unknown / Ambiguous Source Evaluation (fallback path verification)

Outputs detailed metrics by source, by class, latency, and memory footprints.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.finrisk_model_improvements.routing_prototype.router import SourceAwareSentimentRouter
from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer
from tests.evaluation.annotated_news_sample import ANNOTATED_NEWS_SAMPLES

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PREV_EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "finbert_finetuning")
SPLITS_DIR = os.path.join(PREV_EXP_DIR, "data", "splits")

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

    return {
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(f1_sum / len(LABELS), 4),
        "macro_precision": round(p_sum / len(LABELS), 4),
        "macro_recall": round(r_sum / len(LABELS), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def main():
    print("=" * 80)
    print("BENCHMARK: SOURCE-AWARE ROUTING PROTOTYPE VS ORIGINAL FINBERT")
    print("=" * 80)

    # 1. Load Data
    with open(os.path.join(SPLITS_DIR, "test.jsonl"), "r", encoding="utf-8") as f:
        pb_test = [json.loads(line) for line in f if line.strip()]

    pb_texts = [d["text"] for d in pb_test]
    pb_golds = [d["label"] for d in pb_test]
    pb_sources = ["financial_phrasebank"] * len(pb_test)

    # Regulatory benchmark with actual source tags
    reg_texts = []
    reg_golds = []
    reg_sources = []
    for s in ANNOTATED_NEWS_SAMPLES:
        reg_texts.append(s.headline + ". " + s.text)
        reg_golds.append(s.gold_sentiment)
        # Identify source
        hl_low = (s.headline + " " + s.text).lower()
        if "federal reserve" in hl_low or "fomc" in hl_low:
            reg_sources.append("U.S. Federal Reserve Press Releases")
        elif "sec" in hl_low or "securities and exchange" in hl_low:
            reg_sources.append("U.S. SEC Press Releases")
        elif "ecb" in hl_low or "european central bank" in hl_low or "vujčić" in hl_low or "schnabel" in hl_low:
            reg_sources.append("European Central Bank Press Releases")
        else:
            reg_sources.append("U.S. Federal Reserve Press Releases")

    print(f"Loaded {len(pb_texts)} PhraseBank Test samples and {len(reg_texts)} Regulatory Benchmark samples.")

    # 2. Initialize Models
    print("\nInitializing Models...")
    router = SourceAwareSentimentRouter()
    original_finbert = FinBERTSentimentAnalyzer(allow_offline_fallback=False)

    # 3. Evaluate Original FinBERT
    print("\n[Evaluating Original ProsusAI/finbert (Baseline)]...")
    t0 = time.perf_counter()
    pb_preds_orig = [original_finbert.analyze(t).label for t in pb_texts]
    lat_pb_orig = (time.perf_counter() - t0) / len(pb_texts) * 1000.0

    t0 = time.perf_counter()
    reg_preds_orig = [original_finbert.analyze(t).label for t in reg_texts]
    lat_reg_orig = (time.perf_counter() - t0) / len(reg_texts) * 1000.0

    # 4. Evaluate Router Prototype
    print("\n[Evaluating SourceAwareSentimentRouter Prototype]...")
    t0 = time.perf_counter()
    pb_preds_router = [router.analyze(t, source=s).label for t, s in zip(pb_texts, pb_sources)]
    lat_pb_router = (time.perf_counter() - t0) / len(pb_texts) * 1000.0

    t0 = time.perf_counter()
    reg_preds_router = [router.analyze(t, source=s).label for t, s in zip(reg_texts, reg_sources)]
    lat_reg_router = (time.perf_counter() - t0) / len(reg_texts) * 1000.0

    # Fallback test: evaluate on regulatory texts with unknown/omitted source
    reg_preds_fallback = [router.analyze(t, source="unknown_source_feed").label for t in reg_texts]

    # Metrics
    m_pb_orig = compute_metrics(pb_golds, pb_preds_orig)
    m_reg_orig = compute_metrics(reg_golds, reg_preds_orig)

    m_pb_router = compute_metrics(pb_golds, pb_preds_router)
    m_reg_router = compute_metrics(reg_golds, reg_preds_router)
    m_reg_fallback = compute_metrics(reg_golds, reg_preds_fallback)

    # Combined metrics
    comb_golds = pb_golds + reg_golds
    comb_preds_orig = pb_preds_orig + reg_preds_orig
    comb_preds_router = pb_preds_router + reg_preds_router

    m_comb_orig = compute_metrics(comb_golds, comb_preds_orig)
    m_comb_router = compute_metrics(comb_golds, comb_preds_router)

    results = {
        "original_finbert": {
            "phrasebank_test": m_pb_orig,
            "regulatory_benchmark": m_reg_orig,
            "combined": m_comb_orig,
            "latency_ms": round(lat_pb_orig, 2),
        },
        "source_aware_router": {
            "phrasebank_test": m_pb_router,
            "regulatory_benchmark": m_reg_router,
            "regulatory_via_unknown_fallback": m_reg_fallback,
            "combined": m_comb_router,
            "latency_ms": round(lat_pb_router, 2),
        },
        "resource_comparison": {
            "original_finbert": {
                "parameters": 109484547,
                "disk_mb": 418.35,
                "resident_models_in_memory": 1,
            },
            "source_aware_router": {
                "parameters": 218969094,
                "disk_mb": 836.70,
                "resident_models_in_memory": 2,
            },
        },
    }

    out_file = os.path.join(EXP_DIR, "benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"PhraseBank Test (Corporate, N={len(pb_texts)}):")
    print(f"  Original FinBERT: Acc={m_pb_orig['accuracy']:.4f}, Macro-F1={m_pb_orig['macro_f1']:.4f}, Neg Recall={m_pb_orig['per_class']['negative']['recall']:.4f}")
    print(f"  Routing Router:   Acc={m_pb_router['accuracy']:.4f}, Macro-F1={m_pb_router['macro_f1']:.4f}, Neg Recall={m_pb_router['per_class']['negative']['recall']:.4f}")
    print(f"\nRegulatory Benchmark (Central Bank / Enforcement, N={len(reg_texts)}):")
    print(f"  Original FinBERT: Acc={m_reg_orig['accuracy']:.4f}, Macro-F1={m_reg_orig['macro_f1']:.4f}, Neg Recall={m_reg_orig['per_class']['negative']['recall']:.4f}")
    print(f"  Routing Router:   Acc={m_reg_router['accuracy']:.4f}, Macro-F1={m_reg_router['macro_f1']:.4f}, Neg Recall={m_reg_router['per_class']['negative']['recall']:.4f}")
    print(f"  Unknown Fallback: Acc={m_reg_fallback['accuracy']:.4f}, Macro-F1={m_reg_fallback['macro_f1']:.4f}, Neg Recall={m_reg_fallback['per_class']['negative']['recall']:.4f}")
    print(f"\nCombined (N={len(comb_golds)}):")
    print(f"  Original FinBERT: Acc={m_comb_orig['accuracy']:.4f}, Macro-F1={m_comb_orig['macro_f1']:.4f}, Neg Recall={m_comb_orig['per_class']['negative']['recall']:.4f}")
    print(f"  Routing Router:   Acc={m_comb_router['accuracy']:.4f}, Macro-F1={m_comb_router['macro_f1']:.4f}, Neg Recall={m_comb_router['per_class']['negative']['recall']:.4f}")
    print(f"\nLatency: Original={lat_pb_orig:.2f} ms/sample | Router={lat_pb_router:.2f} ms/sample")
    print(f"Results written to: {out_file}")


if __name__ == "__main__":
    main()
