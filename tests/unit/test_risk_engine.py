"""Unit tests for the integrated RiskEngine module."""

from __future__ import annotations

import pytest

from finrisk.nlp.event_classifier import RuleBasedEventClassifier
from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer
from finrisk.risk_engine.engine import RISK_SIGNAL_FIELDS, RiskEngine


def get_mock_engine() -> RiskEngine:
    def mock_pred(text: str) -> dict[str, float]:
        if "default" in text.lower():
            return {"positive": 0.05, "negative": 0.90, "neutral": 0.05}
        return {"positive": 0.70, "negative": 0.10, "neutral": 0.20}

    analyzer = FinBERTSentimentAnalyzer(
        model_name="ProsusAI/finbert-test",
        predictor=mock_pred,
    )
    return RiskEngine(sentiment_analyzer=analyzer, event_classifier=RuleBasedEventClassifier())


def test_process_record_full():
    engine = get_mock_engine()
    raw = {
        "record_id": "rec_001",
        "source": "Financial Wire",
        "published_at": "2024-10-10T12:00:00Z",
        "text": "AlphaCorp defaulted on $20 million debt obligations.",
        "company": "AlphaCorp",
        "source_type": "news",
    }
    sig = engine.process_record(raw)
    assert sig is not None
    for field in RISK_SIGNAL_FIELDS:
        assert field in sig

    assert sig["record_id"] == "rec_001"
    assert sig["company"] == "AlphaCorp"
    assert sig["event_type"] == "Credit Event"
    assert sig["sentiment_label"] == "negative"
    assert sig["sentiment_score"] == -0.85
    assert isinstance(sig["impact_score"], int)
    assert 1 <= sig["impact_score"] <= 10


def test_no_fabrication_of_missing_company_or_date():
    engine = get_mock_engine()
    raw = {
        "record_id": "rec_002",
        "source": "Market Feed",
        "published_at": None,
        "text": "Central bank announces rate decision.",
        "company": None,
        "source_type": "news",
    }
    sig = engine.process_record(raw)
    assert sig is not None
    assert sig["company"] is None
    assert sig["published_at"] is None


def test_skip_empty_text_or_missing_id():
    engine = get_mock_engine()

    # Missing text
    assert engine.process_record({"record_id": "r1", "text": ""}) is None
    assert engine.process_record({"record_id": "r1", "text": "   "}) is None

    # Missing record_id
    assert engine.process_record({"record_id": "", "text": "Valid financial text."}) is None
    assert engine.process_record({"text": "Valid financial text."}) is None

    # Non-dict
    assert engine.process_record("invalid string input") is None  # type: ignore


def test_process_records_batch():
    engine = get_mock_engine()
    records = [
        {"record_id": "r1", "text": "Company default on bonds.", "source": "News"},
        {"record_id": "r2", "text": "   ", "source": "News"},  # Invalid
        {"record_id": "r3", "text": "Profits rise 15 percent.", "source": "News"},
    ]
    signals = engine.process_records(records)
    assert len(signals) == 2
    assert signals[0]["record_id"] == "r1"
    assert signals[1]["record_id"] == "r3"
