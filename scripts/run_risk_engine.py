"""Batch execution runner for Milestone 2: AI/NLP Risk Engine.

Loads normalized records from data/processed/ingested_records.csv, computes
sentiment analysis, event classification, and impact scores, and writes
structured risk signals to data/processed/risk_signals.csv.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finrisk.nlp.event_classifier import RuleBasedEventClassifier
from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer
from finrisk.risk_engine.engine import RISK_SIGNAL_FIELDS, RiskEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_risk_engine")


def sample_offline_finbert_predictor(text: str) -> Dict[str, float]:
    """Deterministic offline mock predictor simulating FinBERT output probabilities.

    Used for local testing and batch verification without downloading 500MB weights.
    Explicitly labeled in outputs as an offline injected predictor.
    """
    lower = text.lower()
    negative_stems = ("delay", "strike", "fine", "inquiry", "provision", "widening", "downgrade", "loss", "risk")
    positive_stems = ("exceed", "above", "rebound", "efficient", "gain", "profit", "bullish")

    neg_matches = sum(1 for stem in negative_stems if stem in lower)
    pos_matches = sum(1 for stem in positive_stems if stem in lower)

    if neg_matches > pos_matches:
        p_neg = min(0.92, 0.60 + 0.10 * neg_matches)
        p_pos = 0.05
        p_neu = round(1.0 - (p_neg + p_pos), 4)
        return {"positive": p_pos, "negative": p_neg, "neutral": p_neu}
    elif pos_matches > neg_matches:
        p_pos = min(0.92, 0.60 + 0.10 * pos_matches)
        p_neg = 0.05
        p_neu = round(1.0 - (p_pos + p_neg), 4)
        return {"positive": p_pos, "negative": p_neg, "neutral": p_neu}
    else:
        return {"positive": 0.10, "negative": 0.10, "neutral": 0.80}


def run_batch_risk_engine(
    input_csv: Path,
    output_csv: Path,
) -> list[dict]:
    """Execute risk engine batch processing and output summary report."""
    print("=" * 65)
    print("FINRISK INTELLIGENCE: RISK ENGINE BATCH RUN (MILESTONE 2)")
    print("=" * 65)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}. Run ingestion first.")

    print(f"\n[1] Loading normalized records from: {input_csv}")

    # Initialize sentiment analyzer with explicit offline predictor
    analyzer = FinBERTSentimentAnalyzer(
        model_name="ProsusAI/finbert (Offline Test Mode)",
        predictor=sample_offline_finbert_predictor,
    )
    classifier = RuleBasedEventClassifier()
    engine = RiskEngine(sentiment_analyzer=analyzer, event_classifier=classifier)

    # Process records
    signals = engine.process_csv(input_csv=input_csv, output_csv=output_csv)
    print(f"\n[2] Processed {len(signals)} risk signals. Saved to: {output_csv}")

    # Summary Statistics
    event_counts: dict[str, int] = {}
    sentiment_counts: dict[str, int] = {}
    impact_scores: list[int] = []

    for sig in signals:
        ev = sig["event_type"]
        sl = sig["sentiment_label"]
        im = sig["impact_score"]
        event_counts[ev] = event_counts.get(ev, 0) + 1
        sentiment_counts[sl] = sentiment_counts.get(sl, 0) + 1
        impact_scores.append(im)

    avg_impact = round(sum(impact_scores) / len(impact_scores), 2) if impact_scores else 0.0

    print("\n" + "=" * 65)
    print("RISK SIGNALS SUMMARY")
    print("=" * 65)
    print(f"  Total Signals Generated: {len(signals)}")
    print(f"  Average Impact Score:    {avg_impact} / 10")
    print("\n  Event Category Distribution:")
    for cat, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {cat:22s}: {count}")
    print("\n  Sentiment Class Distribution:")
    for label, count in sorted(sentiment_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {label:12s}: {count}")

    print("\n" + "=" * 65)
    print("SAMPLE GENERATED RISK SIGNALS (FIRST 2 RECORDS)")
    print("=" * 65)
    for idx, sig in enumerate(signals[:2], 1):
        print(f"\n--- Signal #{idx} (ID: {sig['record_id']}) ---")
        print(f"  Company:      {sig['company'] or '[None]'}")
        print(f"  Event Type:   {sig['event_type']}")
        print(f"  Sentiment:    {sig['sentiment_label']} (Score: {sig['sentiment_score']:.2f}, Conf: {sig['sentiment_confidence']:.2f})")
        print(f"  Impact Score: {sig['impact_score']} / 10")
        print(f"  Rationale:    {sig['scoring_rationale']}")
        print(f"  Text:         {sig['text'][:120]}...")
    print("=" * 65)

    return signals


if __name__ == "__main__":
    in_file = PROJECT_ROOT / "data" / "processed" / "ingested_records.csv"
    out_file = PROJECT_ROOT / "data" / "processed" / "risk_signals.csv"
    run_batch_risk_engine(input_csv=in_file, output_csv=out_file)
