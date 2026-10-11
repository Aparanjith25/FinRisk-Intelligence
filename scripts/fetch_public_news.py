"""Runner script for Milestone 4: Real Public Financial Data Ingestion & Risk Scoring.

Fetches genuine, public-domain financial news feeds (Federal Reserve, SEC, ECB),
normalizes them with full provenance metadata, saves to data/processed/public_news_records.csv,
and generates structured risk signals in data/processed/public_risk_signals.csv.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finrisk.ingestion.public_news import fetch_public_news, save_public_news_csv
from finrisk.nlp.event_classifier import RuleBasedEventClassifier
from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer
from finrisk.risk_engine.engine import RiskEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("fetch_public_news")


def sample_offline_finbert_predictor(text: str) -> dict[str, float]:
    """Deterministic offline mock predictor simulating FinBERT output probabilities.

    Used when PyTorch/Transformers model weights are not loaded locally.
    Explicitly labeled in outputs as an offline injected predictor.
    """
    lower = text.lower()
    negative_stems = ("fine", "penalty", "charge", "fraud", "defrauding", "cease", "desist", "investigation", "recession", "inflation", "hike", "cut", "defaults", "loss")
    positive_stems = ("expand", "grow", "growth", "solid", "rebound", "boost", "resilient", "surplus")

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


def run_public_news_pipeline(
    config_path: Path,
    news_csv_path: Path,
    signals_csv_path: Path,
    timeout: int = 10,
) -> dict:
    """Execute end-to-end public news ingestion and risk scoring."""
    print("=" * 65)
    print("FINRISK INTELLIGENCE: PUBLIC FINANCIAL NEWS INGESTION (MILESTONE 4)")
    print("=" * 65)

    # 1. Ingest public feeds
    print(f"\n[1] Fetching configured public feeds from: {config_path}")
    result = fetch_public_news(config_path=config_path, timeout=timeout)

    print("\n    Feed Status Summary:")
    for status in result.feed_statuses:
        status_symbol = "[OK]" if status.success else "[FAIL]"
        print(f"      {status_symbol:6s} {status.name:38s}: {status.article_count} articles (HTTP {status.status_code})")
        if not status.success:
            print(f"             Error: {status.error_message}")

    print(f"\n    Raw Articles Fetched:     {result.total_raw_count}")
    print(f"    Duplicates Removed:       {result.duplicates_removed}")
    print(f"    Clean Records Accepted:   {result.output_count}")

    if not result.accepted_records:
        print("\n[!] No articles were successfully retrieved. Check internet connection or feed status.")
        return result.to_dict()

    # 2. Save public news records
    save_public_news_csv(result.accepted_records, news_csv_path)
    print(f"\n[2] Saved clean public articles to: {news_csv_path}")

    # 3. Process through RiskEngine
    print(f"\n[3] Generating risk signals for {len(result.accepted_records)} public articles...")
    analyzer = FinBERTSentimentAnalyzer()
    classifier = RuleBasedEventClassifier()
    engine = RiskEngine(sentiment_analyzer=analyzer, event_classifier=classifier)

    signals = engine.process_csv(input_csv=news_csv_path, output_csv=signals_csv_path)
    print(f"    Saved {len(signals)} public risk signals to: {signals_csv_path}")

    # 4. Summary
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
    print("PUBLIC NEWS RISK SIGNALS SUMMARY")
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
    print("SAMPLE REAL PUBLIC ARTICLE RISK SIGNAL")
    print("=" * 65)
    if signals:
        sample = signals[0]
        print(f"  ID:          {sample['record_id']}")
        print(f"  Source:      {sample['source']}")
        print(f"  Published:   {sample['published_at']}")
        print(f"  URL:         {sample['url']}")
        print(f"  Event:       {sample['event_type']}")
        print(f"  Sentiment:   {sample['sentiment_label']} (Score: {sample['sentiment_score']:.2f})")
        print(f"  Impact:      {sample['impact_score']}/10")
        print(f"  Rationale:   {sample['scoring_rationale']}")
        print(f"  Text Excerpt:{sample['text'][:120]}...")
    print("=" * 65)

    return {
        "ingestion_result": result.to_dict(),
        "signals_count": len(signals),
    }


if __name__ == "__main__":
    cfg_file = PROJECT_ROOT / "configs" / "public_feeds.yaml"
    news_out = PROJECT_ROOT / "data" / "processed" / "public_news_records.csv"
    signals_out = PROJECT_ROOT / "data" / "processed" / "public_risk_signals.csv"

    run_public_news_pipeline(
        config_path=cfg_file,
        news_csv_path=news_out,
        signals_csv_path=signals_out,
    )
