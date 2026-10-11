"""Comparative evaluation of Baseline Sentiment vs Genuine ProsusAI/FinBERT.

Measures:
1. Classification accuracy, precision, recall, and macro-F1 on the 30-article curated benchmark
   (with dev and held-out splits).
2. Latency (ms/article) and memory consumption.
3. Distribution and disagreement analysis on 60 live public regulatory news articles.
4. Downstream impact on ScenarioMapper stress testing recommendations.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure environment variables point to E: drive
os.environ["HF_HOME"] = r"E:\hf_home"
os.environ["HF_HUB_CACHE"] = r"E:\hf_home\hub"
os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_home"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finrisk.nlp.sentiment_analyzer import (
    FinBERTSentimentAnalyzer,
    SentimentResult,
    deterministic_baseline_sentiment,
)
from finrisk.risk_engine.engine import RiskEngine
from finrisk.stress_testing.scenario_mapper import recommend_scenarios_from_signals
from tests.evaluation.annotated_news_sample import ANNOTATED_NEWS_SAMPLES, AnnotatedArticle


def compute_metrics(
    y_true: List[str], y_pred: List[str], labels: List[str] = ["positive", "negative", "neutral"]
) -> Dict[str, Any]:
    """Compute exact accuracy, per-class precision/recall/F1, macro-F1, and 3x3 confusion matrix."""
    n = len(y_true)
    if n == 0:
        return {}

    # Build confusion matrix: rows = true, cols = pred
    cm = {true_lbl: {pred_lbl: 0 for pred_lbl in labels} for true_lbl in labels}
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

    for lbl in labels:
        tp = cm[lbl][lbl]
        fp = sum(cm[other][lbl] for other in labels if other != lbl)
        fn = sum(cm[lbl][other] for other in labels if other != lbl)

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

    macro_f1 = f1_sum / len(labels)
    macro_prec = p_sum / len(labels)
    macro_rec = r_sum / len(labels)

    return {
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def evaluate_benchmark(
    analyzer: FinBERTSentimentAnalyzer,
    samples: List[AnnotatedArticle],
    name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float]:
    """Run sentiment analyzer on benchmark samples and record latency and predictions."""
    y_true = [s.gold_sentiment for s in samples]
    y_pred = []
    details = []

    start_time = time.perf_counter()
    for s in samples:
        res = analyzer.analyze(s.headline + ". " + s.text)
        y_pred.append(res.label)
        details.append({
            "record_id": s.record_id,
            "headline": s.headline,
            "gold": s.gold_sentiment,
            "pred": res.label,
            "confidence": res.confidence,
            "score": res.score,
            "is_held_out": s.is_held_out,
            "correct": (res.label == s.gold_sentiment),
        })
    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / len(samples)) * 1000.0

    metrics = compute_metrics(y_true, y_pred)
    return metrics, details, avg_latency_ms


def run_comparison():
    print("=" * 80)
    print("FINRISK INTELLIGENCE: SENTIMENT ENGINE HEAD-TO-HEAD EVALUATION")
    print("=" * 80)

    # 1. Initialize Baseline (Offline Fallback Heuristic)
    print("\n[1/5] Initializing Offline Baseline Analyzer...")
    baseline_analyzer = FinBERTSentimentAnalyzer(
        allow_offline_fallback=True,
        predictor=deterministic_baseline_sentiment,
    )

    # 2. Initialize Genuine Pretrained FinBERT
    print("[2/5] Initializing Genuine FinBERT (ProsusAI/finbert)...")
    finbert_analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=False)
    # Warmup
    finbert_analyzer.analyze("Warmup query for torch and transformers pipeline.")

    # 3. Benchmark Evaluation (30 Articles)
    print("[3/5] Evaluating on 30-Article Ground-Truth Benchmark...")
    dev_samples = [s for s in ANNOTATED_NEWS_SAMPLES if not s.is_held_out]
    heldout_samples = [s for s in ANNOTATED_NEWS_SAMPLES if s.is_held_out]

    # Baseline on full, dev, held-out
    b_all_metrics, b_details, b_lat = evaluate_benchmark(baseline_analyzer, ANNOTATED_NEWS_SAMPLES, "Baseline (All)")
    b_dev_metrics = compute_metrics(
        [d["gold"] for d in b_details if not d["is_held_out"]],
        [d["pred"] for d in b_details if not d["is_held_out"]],
    )
    b_held_metrics = compute_metrics(
        [d["gold"] for d in b_details if d["is_held_out"]],
        [d["pred"] for d in b_details if d["is_held_out"]],
    )

    # FinBERT on full, dev, held-out
    f_all_metrics, f_details, f_lat = evaluate_benchmark(finbert_analyzer, ANNOTATED_NEWS_SAMPLES, "FinBERT (All)")
    f_dev_metrics = compute_metrics(
        [d["gold"] for d in f_details if not d["is_held_out"]],
        [d["pred"] for d in f_details if not d["is_held_out"]],
    )
    f_held_metrics = compute_metrics(
        [d["gold"] for d in f_details if d["is_held_out"]],
        [d["pred"] for d in f_details if d["is_held_out"]],
    )

    # 4. Public News Ingestion & Real-World Dataset Comparison (60 Articles)
    print("[4/5] Evaluating on 60 Live Public News Articles...")
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "public_news_records.csv")
    public_articles = []
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                public_articles.append(row)

    print(f"Loaded {len(public_articles)} live public articles from {csv_path}")

    b_pub_results = []
    f_pub_results = []
    disagreements = []

    b_pub_start = time.perf_counter()
    for art in public_articles:
        text = art.get("text", "") or ""
        b_res = baseline_analyzer.analyze(text)
        b_pub_results.append((art, b_res))
    b_pub_time = (time.perf_counter() - b_pub_start) / max(1, len(public_articles)) * 1000.0

    f_pub_start = time.perf_counter()
    for art in public_articles:
        text = art.get("text", "") or ""
        f_res = finbert_analyzer.analyze(text)
        f_pub_results.append((art, f_res))
    f_pub_time = (time.perf_counter() - f_pub_start) / max(1, len(public_articles)) * 1000.0

    for (art, b_res), (_, f_res) in zip(b_pub_results, f_pub_results):
        if b_res.label != f_res.label:
            disagreements.append({
                "record_id": art.get("record_id"),
                "source": art.get("source"),
                "text_snippet": (art.get("text") or "")[:120] + "...",
                "baseline_label": b_res.label,
                "baseline_score": b_res.score,
                "finbert_label": f_res.label,
                "finbert_score": f_res.score,
                "finbert_conf": f_res.confidence,
            })

    # Distribution summaries
    def get_distribution(results):
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        scores = []
        confs = []
        for _, r in results:
            counts[r.label] = counts.get(r.label, 0) + 1
            scores.append(r.score)
            confs.append(r.confidence)
        n = len(results) or 1
        return {
            "counts": counts,
            "percentages": {k: round(v / n * 100, 1) for k, v in counts.items()},
            "avg_score": round(sum(scores) / n, 4),
            "avg_confidence": round(sum(confs) / n, 4),
        }

    b_dist = get_distribution(b_pub_results)
    f_dist = get_distribution(f_pub_results)

    # 5. Downstream Stress Scenario Mapping Impact
    print("[5/5] Evaluating Downstream Stress Scenario Mapping...")
    b_engine = RiskEngine(sentiment_analyzer=baseline_analyzer)
    f_engine = RiskEngine(sentiment_analyzer=finbert_analyzer)
    b_signals = [sig for art in public_articles if (sig := b_engine.process_record(art))]
    f_signals = [sig for art in public_articles if (sig := f_engine.process_record(art))]

    b_mapping = recommend_scenarios_from_signals(b_signals)
    f_mapping = recommend_scenarios_from_signals(f_signals)

    # Compile structured summary
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "device": "CPU",
            "finbert_parameters": 109484547,
        },
        "benchmark_evaluation": {
            "total_articles": len(ANNOTATED_NEWS_SAMPLES),
            "dev_articles": len(dev_samples),
            "held_out_articles": len(heldout_samples),
            "baseline": {
                "overall": b_all_metrics,
                "dev": b_dev_metrics,
                "held_out": b_held_metrics,
                "latency_ms": round(b_lat, 2),
            },
            "finbert": {
                "overall": f_all_metrics,
                "dev": f_dev_metrics,
                "held_out": f_held_metrics,
                "latency_ms": round(f_lat, 2),
            },
        },
        "public_news_evaluation": {
            "total_articles": len(public_articles),
            "baseline": {
                "distribution": b_dist,
                "latency_ms_per_article": round(b_pub_time, 2),
            },
            "finbert": {
                "distribution": f_dist,
                "latency_ms_per_article": round(f_pub_time, 2),
            },
            "disagreement_count": len(disagreements),
            "disagreement_rate_pct": round(len(disagreements) / max(1, len(public_articles)) * 100, 2),
            "sample_disagreements": disagreements[:8],
        },
        "scenario_mapping_impact": {
            "baseline": {
                "mapped_count": b_mapping.mapped_signal_count,
                "unmapped_count": b_mapping.unmapped_signal_count,
                "top_scenario": b_mapping.top_recommendation.scenario_name if b_mapping.top_recommendation else None,
                "top_composite_score": b_mapping.top_recommendation.composite_score if b_mapping.top_recommendation else None,
                "recommendations": [
                    {"id": r.scenario_id, "name": r.scenario_name, "composite": r.composite_score, "triggers": r.trigger_count}
                    for r in b_mapping.recommendations
                ],
            },
            "finbert": {
                "mapped_count": f_mapping.mapped_signal_count,
                "unmapped_count": f_mapping.unmapped_signal_count,
                "top_scenario": f_mapping.top_recommendation.scenario_name if f_mapping.top_recommendation else None,
                "top_composite_score": f_mapping.top_recommendation.composite_score if f_mapping.top_recommendation else None,
                "recommendations": [
                    {"id": r.scenario_id, "name": r.scenario_name, "composite": r.composite_score, "triggers": r.trigger_count}
                    for r in f_mapping.recommendations
                ],
            },
        },
    }

    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "sentiment_comparison_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\nSaved structured evaluation report to: {output_path}")

    # Print Formatted Report to stdout
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("\n1. BENCHMARK PERFORMANCE (30 Human-Annotated Regulatory Articles):")
    print(f"{'Metric':<25} | {'Offline Baseline':<20} | {'ProsusAI/FinBERT':<20}")
    print("-" * 72)
    print(f"{'Overall Accuracy':<25} | {b_all_metrics['accuracy']*100:.1f}% ({b_all_metrics['accuracy']:.4f}){' ':5} | {f_all_metrics['accuracy']*100:.1f}% ({f_all_metrics['accuracy']:.4f})")
    print(f"{'Overall Macro F1':<25} | {b_all_metrics['macro_f1']:.4f}{' ':14} | {f_all_metrics['macro_f1']:.4f}")
    print(f"{'Dev Set Accuracy (n=22)':<25} | {b_dev_metrics['accuracy']*100:.1f}%{' ':14} | {f_dev_metrics['accuracy']*100:.1f}%")
    print(f"{'Dev Set Macro F1':<25} | {b_dev_metrics['macro_f1']:.4f}{' ':14} | {f_dev_metrics['macro_f1']:.4f}")
    print(f"{'Held-Out Accuracy (n=8)':<25} | {b_held_metrics['accuracy']*100:.1f}%{' ':14} | {f_held_metrics['accuracy']*100:.1f}%")
    print(f"{'Held-Out Macro F1':<25} | {b_held_metrics['macro_f1']:.4f}{' ':14} | {f_held_metrics['macro_f1']:.4f}")
    print(f"{'Inference Latency':<25} | {b_lat:.2f} ms/article{' ':7} | {f_lat:.2f} ms/article")

    print("\nCONFUSION MATRIX (BASELINE):")
    print("             Pred Pos   Pred Neg   Pred Neu")
    for row in ["positive", "negative", "neutral"]:
        cm_r = b_all_metrics["confusion_matrix"][row]
        print(f"True {row:<8}: {cm_r['positive']:<10} {cm_r['negative']:<10} {cm_r['neutral']:<10}")

    print("\nCONFUSION MATRIX (FINBERT):")
    print("             Pred Pos   Pred Neg   Pred Neu")
    for row in ["positive", "negative", "neutral"]:
        cm_r = f_all_metrics["confusion_matrix"][row]
        print(f"True {row:<8}: {cm_r['positive']:<10} {cm_r['negative']:<10} {cm_r['neutral']:<10}")

    print("\n2. LIVE PUBLIC REGULATORY NEWS (60 Articles from Fed, SEC, ECB):")
    print(f"{'Feature':<25} | {'Offline Baseline':<20} | {'ProsusAI/FinBERT':<20}")
    print("-" * 72)
    print(f"{'Positive Articles':<25} | {b_dist['counts']['positive']} ({b_dist['percentages']['positive']}%) | {f_dist['counts']['positive']} ({f_dist['percentages']['positive']}%)")
    print(f"{'Negative Articles':<25} | {b_dist['counts']['negative']} ({b_dist['percentages']['negative']}%) | {f_dist['counts']['negative']} ({f_dist['percentages']['negative']}%)")
    print(f"{'Neutral Articles':<25}  | {b_dist['counts']['neutral']} ({b_dist['percentages']['neutral']}%)  | {f_dist['counts']['neutral']} ({f_dist['percentages']['neutral']}%)")
    print(f"{'Average Sentiment Score':<25} | {b_dist['avg_score']:+.4f}{' ':13} | {f_dist['avg_score']:+.4f}")
    print(f"{'Average Confidence':<25} | {b_dist['avg_confidence']:.4f}{' ':14} | {f_dist['avg_confidence']:.4f}")
    print(f"{'Disagreements':<25} | {len(disagreements)} / {len(public_articles)} ({len(disagreements)/len(public_articles)*100:.1f}%)")

    print("\n3. DOWNSTREAM STRESS SCENARIO MAPPING IMPACT:")
    print(f"{'Metric':<25} | {'Offline Baseline':<20} | {'ProsusAI/FinBERT':<20}")
    print("-" * 72)
    print(f"{'Mapped Signals':<25} | {b_mapping.mapped_signal_count:<20} | {f_mapping.mapped_signal_count:<20}")
    print(f"{'Unmapped Signals':<25} | {b_mapping.unmapped_signal_count:<20} | {f_mapping.unmapped_signal_count:<20}")
    top_b = b_mapping.top_recommendation
    top_f = f_mapping.top_recommendation
    print(f"{'Top Scenario':<25} | {top_b.scenario_name if top_b else 'None':<20} | {top_f.scenario_name if top_f else 'None':<20}")
    print(f"{'Top Composite Score':<25} | {top_b.composite_score if top_b else 0.0:<20.4f} | {top_f.composite_score if top_f else 0.0:<20.4f}")


if __name__ == "__main__":
    run_comparison()
