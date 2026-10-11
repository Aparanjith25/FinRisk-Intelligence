"""Core Risk Engine module.

Integrates normalized data ingestion records with financial sentiment analysis,
event classification, and explainable impact scoring to produce structured risk signals.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from finrisk.nlp.event_classifier import BaseEventClassifier, RuleBasedEventClassifier
from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer
from finrisk.risk_engine.impact_scorer import compute_impact_score

logger = logging.getLogger(__name__)

# Standard target schema for downstream risk signals
RISK_SIGNAL_FIELDS = (
    "record_id",
    "source",
    "published_at",
    "text",
    "company",
    "sentiment_score",
    "sentiment_label",
    "sentiment_confidence",
    "event_type",
    "impact_score",
    "scoring_rationale",
    "url",
    "provenance",
    "inference_source",
)


@dataclass
class RiskSignal:
    """Structured financial risk signal for downstream portfolio stress testing."""

    record_id: str
    source: str
    published_at: Optional[str]
    text: str
    company: Optional[str]
    sentiment_score: float
    sentiment_label: str
    sentiment_confidence: float
    event_type: str
    impact_score: int
    scoring_rationale: str
    url: Optional[str] = None
    provenance: str = "synthetic"
    inference_source: str = "offline_injected_mock"

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary matching RISK_SIGNAL_FIELDS."""
        return asdict(self)


class RiskEngine:
    """Integrated engine processing financial text records into quantitative risk signals.

    Ensures zero fabrication of companies, dates, or model confidence scores.
    """

    def __init__(
        self,
        sentiment_analyzer: Optional[FinBERTSentimentAnalyzer] = None,
        event_classifier: Optional[BaseEventClassifier] = None,
    ) -> None:
        """Initialize RiskEngine with modular NLP and classification components.

        Args:
            sentiment_analyzer: Instance of FinBERTSentimentAnalyzer (or one configured with mock predictor).
            event_classifier: Instance of BaseEventClassifier (defaults to RuleBasedEventClassifier).
        """
        self.sentiment_analyzer = sentiment_analyzer or FinBERTSentimentAnalyzer()
        self.event_classifier = event_classifier or RuleBasedEventClassifier()

    def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single normalized record into a structured risk signal.

        Args:
            record: Normalized dictionary from ingestion pipeline.

        Returns:
            Dictionary matching RISK_SIGNAL_FIELDS, or None if record is invalid.
        """
        if not isinstance(record, dict):
            logger.warning("Rejected non-dict input in risk engine: %s", record)
            return None

        record_id = record.get("record_id")
        text = record.get("text")
        source = record.get("source", "Unknown Source")
        published_at = record.get("published_at")
        company = record.get("company")

        # Validate non-empty text and record ID
        if not text or not str(text).strip():
            logger.warning("Skipping record with missing or empty text (ID: %s)", record_id)
            return None

        if not record_id or not str(record_id).strip():
            logger.warning("Skipping record with missing record_id")
            return None

        clean_text = str(text).strip()

        # 1. Financial Sentiment Analysis
        sentiment = self.sentiment_analyzer.analyze(clean_text)

        # 2. Event Classification
        event = self.event_classifier.classify(clean_text)

        # 3. Impact Scoring
        impact = compute_impact_score(
            event_category=event.category,
            sentiment_score=sentiment.score,
            sentiment_confidence=sentiment.confidence,
            company=company,
            text=clean_text,
        )

        url = record.get("url")
        provenance = record.get("provenance") or (
            "real_public_news" if record.get("source_type") == "public_news" else "synthetic"
        )
        inference_source = getattr(sentiment, "inference_source", "offline_injected_mock")

        signal = RiskSignal(
            record_id=str(record_id).strip(),
            source=str(source).strip(),
            published_at=str(published_at).strip() if (published_at and str(published_at).strip()) else None,
            text=clean_text,
            company=str(company).strip() if (company and str(company).strip()) else None,
            sentiment_score=sentiment.score,
            sentiment_label=sentiment.label,
            sentiment_confidence=sentiment.confidence,
            event_type=event.category,
            impact_score=impact.score,
            scoring_rationale=impact.rationale,
            url=str(url).strip() if url else None,
            provenance=str(provenance).strip(),
            inference_source=str(inference_source).strip(),
        )

        return signal.to_dict()

    def process_records(self, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process an iterable of normalized records into a list of structured risk signals."""
        signals: List[Dict[str, Any]] = []
        for r in records:
            sig = self.process_record(r)
            if sig is not None:
                signals.append(sig)
        return signals

    def process_csv(
        self,
        input_csv: Union[str, Path],
        output_csv: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """Load records from input CSV, generate risk signals, and optionally write to output CSV.

        Args:
            input_csv: Path to input CSV (e.g., data/processed/ingested_records.csv).
            output_csv: Optional destination path for risk signals CSV.

        Returns:
            List of generated risk signal dictionaries.
        """
        in_path = Path(input_csv)
        if not in_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {in_path}")

        records: List[Dict[str, Any]] = []
        with open(in_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            records.extend(reader)

        logger.info("Processing %d records from %s through RiskEngine", len(records), in_path)
        signals = self.process_records(records)

        if output_csv:
            out_path = Path(output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RISK_SIGNAL_FIELDS)
                writer.writeheader()
                writer.writerows(signals)
            logger.info("Saved %d risk signals to %s", len(signals), out_path)

        return signals
