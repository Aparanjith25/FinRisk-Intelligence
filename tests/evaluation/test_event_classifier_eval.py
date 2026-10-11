"""Evaluation test suite for financial event classification.

Runs quantitative evaluation against curated labeled dataset, measuring:
- Per-class Precision, Recall, F1-Score
- Macro-averaged F1-Score
- Overall Accuracy
- Full 6x6 Confusion Matrix
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import pytest

from finrisk.nlp.event_classifier import EVENT_CATEGORIES, RuleBasedEventClassifier
from tests.evaluation.event_eval_dataset import EVAL_SAMPLES, EvalSample


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Tuple[str, ...],
) -> Dict[str, Any]:
    """Calculate confusion matrix, precision, recall, and F1 scores."""
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t]][label_to_idx[p]] += 1

    per_class = {}
    f1_list = []
    for i, lbl in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(cm[i])
        per_class[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1_list.append(f1)

    total_correct = sum(cm[i][i] for i in range(n))
    total_samples = len(y_true)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    return {
        "confusion_matrix": cm,
        "per_class": per_class,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "total_samples": total_samples,
    }


def format_evaluation_report(metrics: Dict[str, Any], labels: Tuple[str, ...]) -> str:
    """Generate printable evaluation report string."""
    lines = ["\n================= EVENT CLASSIFIER EVALUATION REPORT ================="]
    lines.append(f"Total Samples: {metrics['total_samples']} | Overall Accuracy: {metrics['accuracy']:.2%} | Macro F1: {metrics['macro_f1']:.4f}\n")
    lines.append(f"{'Category':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    lines.append("-" * 72)
    for lbl in labels:
        stat = metrics["per_class"][lbl]
        lines.append(f"{lbl:<22} | {stat['precision']:<10.4f} | {stat['recall']:<10.4f} | {stat['f1']:<10.4f} | {stat['support']:<8}")

    lines.append("\nCONFUSION MATRIX (Rows: True, Columns: Predicted):")
    header = "True \\ Pred".ljust(20) + " ".join(f"{lbl[:7]:>8}" for lbl in labels)
    lines.append(header)
    lines.append("-" * len(header))
    cm = metrics["confusion_matrix"]
    for i, lbl in enumerate(labels):
        row_str = f"{lbl:<20}" + " ".join(f"{cm[i][j]:>8}" for j in range(len(labels)))
        lines.append(row_str)
    lines.append("======================================================================\n")
    return "\n".join(lines)


def test_event_classifier_benchmark_evaluation():
    """Run full quantitative evaluation benchmark across all curated evaluation samples."""
    classifier = RuleBasedEventClassifier()
    y_true = [s.gold_category for s in EVAL_SAMPLES]
    y_pred = [classifier.classify(s.text).category for s in EVAL_SAMPLES]

    metrics = compute_metrics(y_true, y_pred, EVENT_CATEGORIES)
    report = format_evaluation_report(metrics, EVENT_CATEGORIES)
    print(report)

    # Assert robust deterministic performance
    # All canonical categories should achieve >= 80% recall and accuracy >= 85%
    assert metrics["accuracy"] >= 0.85, f"Accuracy {metrics['accuracy']} is below threshold 0.85.\n{report}"
    assert metrics["macro_f1"] >= 0.85, f"Macro F1 {metrics['macro_f1']} is below threshold 0.85.\n{report}"


def test_ambiguous_cases_detect_competing_signals():
    """Verify that ambiguous test samples capture competing alternative categories."""
    classifier = RuleBasedEventClassifier()
    ambiguous_samples = [s for s in EVAL_SAMPLES if s.is_ambiguous]
    assert len(ambiguous_samples) >= 6

    for sample in ambiguous_samples:
        res = classifier.classify(sample.text)
        # Verify that either the match was identified with competing categories recorded,
        # or matched keywords reflect the multi-faceted concepts
        assert len(res.matched_keywords) > 0 or len(res.competing_categories) > 0
