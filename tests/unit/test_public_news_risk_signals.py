"""Unit tests for processing public financial news through RiskEngine."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from finrisk.ingestion.public_news import FeedConfig, parse_public_feed
from finrisk.nlp.event_classifier import RuleBasedEventClassifier
from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer
from finrisk.risk_engine.engine import RISK_SIGNAL_FIELDS, RiskEngine

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
FED_FIXTURE_PATH = FIXTURES_DIR / "sample_fed_feed.xml"
SEC_FIXTURE_PATH = FIXTURES_DIR / "sample_sec_feed.xml"


@pytest.fixture
def risk_engine() -> RiskEngine:
    def mock_pred(text: str) -> dict[str, float]:
        if any(w in text.lower() for w in ["defraud", "enforcement", "charges", "deficiencies"]):
            return {"positive": 0.05, "negative": 0.85, "neutral": 0.10}
        return {"positive": 0.40, "negative": 0.20, "neutral": 0.40}

    analyzer = FinBERTSentimentAnalyzer(
        model_name="ProsusAI/finbert-test",
        predictor=mock_pred,
    )
    return RiskEngine(
        sentiment_analyzer=analyzer,
        event_classifier=RuleBasedEventClassifier(),
    )


def test_process_public_news_records_provenance_and_metadata(risk_engine):
    """Verify that public news records produce risk signals with strict provenance tracking."""
    xml_content = FED_FIXTURE_PATH.read_text(encoding="utf-8")
    cfg = FeedConfig(
        feed_id="fed_fixture",
        name="U.S. Federal Reserve",
        url="https://www.federalreserve.gov/feeds/press_all.xml",
        description="Fed press releases",
        provenance="real_public_news",
    )
    raw_records = parse_public_feed(xml_content, cfg, retrieved_at="2026-10-10T14:00:00Z")

    # Give them record_ids like normalizer would
    records = []
    for idx, r in enumerate(raw_records, start=1):
        r["record_id"] = f"pub_fed_{idx}"
        records.append(r)

    signals = risk_engine.process_records(records)
    assert len(signals) == 2

    sig = signals[0]
    # Check schema fields completeness
    for field in RISK_SIGNAL_FIELDS:
        assert field in sig, f"Missing expected field: {field}"

    assert sig["provenance"] == "real_public_news"
    assert sig["inference_source"] == "offline_injected_mock"
    assert sig["url"] == "https://www.federalreserve.gov/newsevents/pressreleases/monetary20261001a.htm"
    assert sig["source"] == "U.S. Federal Reserve"
    assert -1.0 <= sig["sentiment_score"] <= 1.0
    assert 1 <= sig["impact_score"] <= 10
    assert len(sig["scoring_rationale"]) > 0
    # Zero fabrication of missing company
    assert sig["company"] is None


def test_process_public_news_csv(risk_engine, tmp_path):
    """Verify RiskEngine.process_csv correctly processes public news CSV files."""
    in_csv = tmp_path / "public_news.csv"
    out_csv = tmp_path / "public_risk_signals.csv"

    sample_records = [
        {
            "record_id": "sec_001",
            "source": "U.S. SEC",
            "published_at": "2026-10-05T10:15:00Z",
            "text": "SEC Charges Commercial Real Estate Issuer with Defrauding Investors.",
            "url": "https://www.sec.gov/newsroom/press-releases/2026-95",
            "company": "Horizon Real Estate Partners",
            "source_type": "public_news",
            "provenance": "real_public_news",
            "retrieved_at": "2026-10-10T14:00:00Z",
        }
    ]

    with open(in_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_records[0].keys()))
        writer.writeheader()
        writer.writerows(sample_records)

    signals = risk_engine.process_csv(in_csv, out_csv)
    assert len(signals) == 1
    assert out_csv.exists()

    with open(out_csv, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["record_id"] == "sec_001"
        assert reader[0]["provenance"] == "real_public_news"
        assert reader[0]["inference_source"] == "offline_injected_mock"
        assert reader[0]["company"] == "Horizon Real Estate Partners"
        assert reader[0]["url"] == "https://www.sec.gov/newsroom/press-releases/2026-95"


def test_provenance_separation_synthetic_vs_public(risk_engine):
    """Verify that synthetic and real public records maintain separate provenance tags."""
    synth_record = {
        "record_id": "synth_001",
        "source": "Synthetic Financial News",
        "published_at": "2026-09-01T00:00:00Z",
        "text": "AlphaCorp reports quarterly revenue drop amid tariff pressures.",
        "company": "AlphaCorp",
        "source_type": "rss",
        "provenance": "synthetic",
    }
    public_record = {
        "record_id": "pub_001",
        "source": "U.S. Federal Reserve",
        "published_at": "2026-10-01T14:00:00Z",
        "text": "Federal Open Market Committee maintains federal funds rate.",
        "company": None,
        "source_type": "public_news",
        "provenance": "real_public_news",
        "url": "https://www.federalreserve.gov/press/1",
    }

    synth_sig = risk_engine.process_record(synth_record)
    pub_sig = risk_engine.process_record(public_record)

    assert synth_sig["provenance"] == "synthetic"
    assert pub_sig["provenance"] == "real_public_news"
    assert synth_sig["provenance"] != pub_sig["provenance"]
