"""Regression tests covering critical public news fetch bug fixes.

Covers:
1. Sentiment inference when torch/transformers are missing and when present.
2. Accurate is_mock and inference_source metadata labeling.
3. End-to-end public news ingestion with offline sentiment fallback and neural inference.
4. Strict exclusion of synthetic rows in Real Public Financial News mode.
5. Explicit empty state when real news data is absent (zero synthetic substitution).
6. Provenance preservation and deduplication in Combined mode.
7. Feed failure fault isolation (partial failure does not abort pipeline).
8. Unexpected errors remain visible and are not silently masked.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from finrisk.dashboard.app import load_local_risk_signals
from finrisk.ingestion.public_news import (
    FeedConfig,
    fetch_public_news,
    parse_public_feed,
)
from finrisk.nlp.sentiment_analyzer import (
    FinBERTSentimentAnalyzer,
    SentimentResult,
    deterministic_baseline_sentiment,
)
from finrisk.risk_engine.engine import RiskEngine

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
FED_FIXTURE_PATH = FIXTURES_DIR / "sample_fed_feed.xml"


def is_torch_available() -> bool:
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


def test_sentiment_inference_environment_modes(monkeypatch):
    """Verify sentiment inference executes appropriately in both environments."""
    if not is_torch_available():
        # Original C-drive environment: torch/transformers absent
        analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=True)
        res = analyzer.analyze("FOMC raised interest rates by 25 basis points amid persistent inflation.")

        assert isinstance(res, SentimentResult)
        assert res.label in ("positive", "negative", "neutral")
        assert res.is_mock is True
        assert res.inference_source == "offline_fallback_baseline"
        assert -1.0 <= res.score <= 1.0
        assert 0.0 <= res.confidence <= 1.0
        assert sum(res.probabilities.values()) == pytest.approx(1.0, abs=0.01)

        # When fallback is explicitly disabled in environment without torch, raises ImportError
        strict_analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=False)
        with pytest.raises(ImportError, match="transformers and torch are required"):
            strict_analyzer.analyze("Some text")
    else:
        # E:\finrisk_ml_env: genuine pretrained FinBERT inference
        analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=True)
        res = analyzer.analyze("FOMC raised interest rates by 25 basis points amid persistent inflation.")

        assert isinstance(res, SentimentResult)
        assert res.label in ("positive", "negative", "neutral")
        assert res.is_mock is False
        assert res.inference_source == "pretrained_transformers"
        assert res.model_name == "ProsusAI/finbert"
        assert -1.0 <= res.score <= 1.0
        assert 0.0 <= res.confidence <= 1.0

        # When fallback is disabled in ML environment, inference succeeds
        strict_analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=False)
        res_strict = strict_analyzer.analyze("FOMC raised interest rates by 25 basis points amid persistent inflation.")
        assert res_strict.is_mock is False
        assert res_strict.inference_source == "pretrained_transformers"

        # Verify fallback triggers as safety net if torch is simulated as absent
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "transformers", None)
        sim_analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=True)
        res_sim = sim_analyzer.analyze("Some text")
        assert res_sim.is_mock is True
        assert res_sim.inference_source == "offline_fallback_baseline"


def test_sentiment_metadata_for_fallback_and_mock_predictors(monkeypatch):
    """Verify inference results carry truthful provenance and mock flags."""
    # 1. Environment-native analyzer metadata
    analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=True)
    res = analyzer.analyze("Federal Reserve orders civil penalties for compliance deficiencies.")
    if not is_torch_available():
        assert res.is_mock is True
        assert res.inference_source == "offline_fallback_baseline"
        assert "Offline Fallback Baseline" in res.model_name
    else:
        assert res.is_mock is False
        assert res.inference_source == "pretrained_transformers"
        assert res.model_name == "ProsusAI/finbert"

    # 2. Simulated offline fallback baseline (verified across all environments)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    fallback_analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=True)
    res_fallback = fallback_analyzer.analyze("Federal Reserve orders civil penalties for compliance deficiencies.")
    assert res_fallback.is_mock is True
    assert res_fallback.inference_source == "offline_fallback_baseline"
    assert "Offline Fallback Baseline" in res_fallback.model_name
    assert res_fallback.inference_source != "pretrained_transformers"

    # 3. User-injected mock (verified across all environments)
    mock_analyzer = FinBERTSentimentAnalyzer(
        predictor=lambda t: {"positive": 0.1, "negative": 0.8, "neutral": 0.1}
    )
    res_mock = mock_analyzer.analyze("Company reports revenue growth.")
    assert res_mock.is_mock is True
    assert res_mock.inference_source == "offline_injected_mock"
    assert "Injected Predictor" in res_mock.model_name


def test_public_news_risk_scoring_end_to_end(tmp_path):
    """Verify public news records can be scored end-to-end in native environment."""
    xml_content = FED_FIXTURE_PATH.read_text(encoding="utf-8")
    cfg = FeedConfig(
        feed_id="fed_test",
        name="U.S. Federal Reserve",
        url="https://www.federalreserve.gov/feeds/press_all.xml",
        description="Fed press releases",
        provenance="real_public_news",
    )
    records = parse_public_feed(xml_content, cfg, retrieved_at="2026-10-10T15:00:00Z")
    for idx, r in enumerate(records, start=1):
        r["record_id"] = f"pub_test_{idx}"

    engine = RiskEngine()
    signals = engine.process_records(records)

    assert len(signals) == 2
    for sig in signals:
        assert sig["provenance"] == "real_public_news"
        assert sig["company"] is None  # Zero entity fabrication
        assert sig["url"] is not None
        assert sig["source"] == "U.S. Federal Reserve"
        if not is_torch_available():
            assert sig["inference_source"] == "offline_fallback_baseline"
        else:
            assert sig["inference_source"] == "pretrained_transformers"


def test_public_news_risk_scoring_with_offline_fallback(monkeypatch, tmp_path):
    """Verify public news records can be scored end-to-end with offline fallback."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    xml_content = FED_FIXTURE_PATH.read_text(encoding="utf-8")
    cfg = FeedConfig(
        feed_id="fed_test",
        name="U.S. Federal Reserve",
        url="https://www.federalreserve.gov/feeds/press_all.xml",
        description="Fed press releases",
        provenance="real_public_news",
    )
    records = parse_public_feed(xml_content, cfg, retrieved_at="2026-10-10T15:00:00Z")
    for idx, r in enumerate(records, start=1):
        r["record_id"] = f"pub_test_{idx}"

    engine = RiskEngine()
    signals = engine.process_records(records)

    assert len(signals) == 2
    for sig in signals:
        assert sig["provenance"] == "real_public_news"
        assert sig["inference_source"] == "offline_fallback_baseline"
        assert sig["company"] is None
        assert sig["url"] is not None
        assert sig["source"] == "U.S. Federal Reserve"


def test_real_public_news_mode_excludes_synthetic_records():
    """Verify Real Public Financial News mode strictly excludes synthetic rows."""
    signals = load_local_risk_signals(provenance="real_public_news")
    assert isinstance(signals, list)
    for sig in signals:
        prov = (sig.get("provenance") or "").strip().lower()
        assert prov == "real_public_news", f"Synthetic record found in real news mode: {sig}"
        assert "synthetic" not in sig.get("source", "").lower()
        assert "synthetic" not in sig.get("url", "").lower()


def test_empty_real_news_dataset_returns_empty_without_synthetic_substitution(monkeypatch, tmp_path):
    """Verify that an empty real news store yields empty list, never synthetic records."""
    empty_public_path = tmp_path / "empty_public_signals.csv"
    empty_public_path.touch()

    monkeypatch.setattr("finrisk.dashboard.app.PUBLIC_RISK_SIGNALS_PATH", empty_public_path)

    signals = load_local_risk_signals(provenance="real_public_news")
    assert signals == [], "Must return empty list when public risk signals are empty"


def test_combined_mode_preserves_provenance_and_deduplicates():
    """Verify Combined mode contains both provenances and eliminates duplicate IDs."""
    combined_signals = load_local_risk_signals(provenance="all")
    assert isinstance(combined_signals, list)

    seen_ids = set()
    has_synth = False
    has_public = False

    for sig in combined_signals:
        rid = sig.get("record_id")
        assert rid not in seen_ids, f"Duplicate record_id in combined view: {rid}"
        seen_ids.add(rid)

        prov = (sig.get("provenance") or "synthetic").strip().lower()
        if prov == "synthetic":
            has_synth = True
        elif prov == "real_public_news":
            has_public = True

    # Both datasets should be represented
    assert has_synth is True
    assert has_public is True


@patch("finrisk.ingestion.public_news.fetch_feed_content")
def test_feed_failure_isolation_does_not_abort_pipeline(mock_fetch):
    """Verify that transient or persistent errors on one feed do not stop remaining feeds."""
    fed_xml = FED_FIXTURE_PATH.read_text(encoding="utf-8")

    def side_effect(url, **kwargs):
        if "federalreserve" in url:
            return fed_xml, 200
        elif "sec.gov" in url:
            raise ConnectionError("Connection reset by peer (504 Gateway Timeout)")
        elif "ecb.europa" in url:
            return "<rss><channel><title>ECB</title></channel></rss>", 200
        return "<rss></rss>", 200

    mock_fetch.side_effect = side_effect

    result = fetch_public_news()
    assert result.output_count >= 2

    sec_status = next(s for s in result.feed_statuses if "sec" in s.feed_id)
    assert sec_status.success is False
    assert "Connection reset by peer" in str(sec_status.error_message)

    fed_status = next(s for s in result.feed_statuses if "federal_reserve" in s.feed_id)
    assert fed_status.success is True
    assert fed_status.article_count == 2


def test_unexpected_exceptions_remain_visible():
    """Verify that unexpected exceptions (such as bad config path) raise and are not masked."""
    with pytest.raises(FileNotFoundError):
        fetch_public_news(config_path="nonexistent_config_path_123.yaml")
