"""Unit tests for financial sentiment analyzer."""

from __future__ import annotations

import pytest

from finrisk.nlp.sentiment_analyzer import (
    FinBERTSentimentAnalyzer,
    compute_continuous_score,
    deterministic_baseline_sentiment,
)


def is_torch_available() -> bool:
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


def test_continuous_score_calculation():
    # Strong positive
    score_pos = compute_continuous_score({"positive": 0.90, "negative": 0.05, "neutral": 0.05})
    assert score_pos == 0.85

    # Strong negative
    score_neg = compute_continuous_score({"positive": 0.02, "negative": 0.88, "neutral": 0.10})
    assert score_neg == -0.86

    # Neutral dominated
    score_neu = compute_continuous_score({"positive": 0.05, "negative": 0.05, "neutral": 0.90})
    assert score_neu == 0.0

    # Bounds check & clamping
    assert compute_continuous_score({"positive": 1.5, "negative": 0.0}) == 1.0
    assert compute_continuous_score({"positive": 0.0, "negative": 1.5}) == -1.0


def test_continuous_score_missing_keys():
    # Missing positive or negative should safely default to 0
    assert compute_continuous_score({}) == 0.0
    assert compute_continuous_score({"positive": 0.7}) == 0.7
    assert compute_continuous_score({"negative": 0.4}) == -0.4


def test_sentiment_analyzer_with_mock_predictor():
    def mock_pred(text: str) -> dict[str, float]:
        if "bullish" in text.lower():
            return {"positive": 0.85, "negative": 0.05, "neutral": 0.10}
        return {"positive": 0.05, "negative": 0.80, "neutral": 0.15}

    analyzer = FinBERTSentimentAnalyzer(
        model_name="ProsusAI/finbert",
        predictor=mock_pred,
    )

    res_pos = analyzer.analyze("Analysts maintain bullish outlook on operating margins.")
    assert res_pos.label == "positive"
    assert res_pos.score == 0.80
    assert res_pos.confidence == 0.85
    assert "Injected Predictor" in res_pos.model_name
    assert res_pos.is_mock is True
    assert res_pos.inference_source == "offline_injected_mock"

    res_neg = analyzer.analyze("Severe credit downgrade announced for subsidiary.")
    assert res_neg.label == "negative"
    assert res_neg.score == -0.75
    assert res_neg.confidence == 0.80
    assert res_neg.is_mock is True


def test_sentiment_analyzer_empty_input_raises():
    analyzer = FinBERTSentimentAnalyzer(
        predictor=lambda t: {"positive": 0.5, "negative": 0.5, "neutral": 0.0}
    )
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        analyzer.analyze("")

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        analyzer.analyze("    ")


def test_sentiment_analyzer_batch():
    analyzer = FinBERTSentimentAnalyzer(
        predictor=lambda t: {"positive": 0.6, "negative": 0.2, "neutral": 0.2}
    )
    results = analyzer.analyze_batch(["Text item 1", "Text item 2"])
    assert len(results) == 2
    assert results[0].label == "positive"
    assert results[1].label == "positive"


def test_deterministic_baseline_sentiment_categories():
    # Positive text with growth / expansion
    pos = deterministic_baseline_sentiment("Company reported solid revenue growth, expansion into new markets, and increased profits.")
    assert pos["positive"] > pos["negative"]
    assert pos["positive"] >= 0.60

    # Negative text with default / loss / penalty
    neg = deterministic_baseline_sentiment("Subsidiary suffered severe loss, debt default, and heavy regulatory penalties.")
    assert neg["negative"] > pos["positive"] or neg["negative"] >= 0.60

    # Neutral text with routine corporate disclosure
    neu = deterministic_baseline_sentiment("The committee held its scheduled quarterly advisory meeting at the headquarters.")
    assert neu["neutral"] == 0.80
    assert neu["positive"] == 0.10
    assert neu["negative"] == 0.10


def test_sentiment_analyzer_environment_contract():
    """Verify sentiment analyzer behavior in the current environment."""
    analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=True)

    # 1. Negative text
    res_neg = analyzer.analyze("Severe recession warnings, rising inflation, and credit downgrades across banking sector.")
    assert -1.0 <= res_neg.score <= 1.0
    assert res_neg.score < 0.0
    assert res_neg.label == "negative"
    assert 0.0 <= res_neg.confidence <= 1.0
    assert "positive" in res_neg.probabilities
    assert "negative" in res_neg.probabilities
    assert "neutral" in res_neg.probabilities

    # 2. Positive text
    res_pos = analyzer.analyze("Record profitable year with unprecedented revenue growth and major credit rating upgrade.")
    assert res_pos.score > 0.0
    assert res_pos.label == "positive"

    # 3. Neutral text
    res_neu = analyzer.analyze("The annual shareholder meeting will be held on Thursday at the hotel convention hall.")
    assert abs(res_neu.score) <= 0.05
    assert res_neu.label == "neutral"

    if not is_torch_available():
        # C-drive baseline environment: fallback expected
        assert res_neg.is_mock is True
        assert res_neg.inference_source == "offline_fallback_baseline"
        assert "Offline Fallback Baseline" in res_neg.model_name
        assert res_pos.is_mock is True
        assert res_pos.inference_source == "offline_fallback_baseline"
    else:
        # E:\finrisk_ml_env environment: genuine pretrained FinBERT expected
        assert res_neg.is_mock is False
        assert res_neg.inference_source == "pretrained_transformers"
        assert res_neg.model_name == "ProsusAI/finbert"
        assert res_pos.is_mock is False
        assert res_pos.inference_source == "pretrained_transformers"


def test_sentiment_analyzer_simulated_offline_fallback(monkeypatch):
    """Verify fallback contract when torch/transformers are simulated as absent."""
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=True)

    res_neg = analyzer.analyze("Severe recession warnings, rising inflation, and credit downgrades across banking sector.")
    assert res_neg.is_mock is True
    assert res_neg.inference_source == "offline_fallback_baseline"
    assert "Offline Fallback Baseline" in res_neg.model_name
    assert res_neg.label == "negative"

    res_pos = analyzer.analyze("Record profitable year with unprecedented revenue growth and major credit rating upgrade.")
    assert res_pos.is_mock is True
    assert res_pos.inference_source == "offline_fallback_baseline"
    assert res_pos.label == "positive"


def test_sentiment_result_to_dict_and_contract():
    analyzer = FinBERTSentimentAnalyzer(
        predictor=lambda t: {"positive": 0.75, "negative": 0.15, "neutral": 0.10}
    )
    res = analyzer.analyze("Solid quarterly revenue.")
    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["score"] == 0.60
    assert d["label"] == "positive"
    assert d["confidence"] == 0.75
    assert d["is_mock"] is True
    assert d["inference_source"] == "offline_injected_mock"
    assert "probabilities" in d


def test_sentiment_unnormalized_probabilities_and_noise():
    # If injected predictor returns unnormalized probabilities, analyzer should normalize
    analyzer = FinBERTSentimentAnalyzer(
        predictor=lambda t: {"positive": 3.0, "negative": 1.0, "neutral": 0.0}
    )
    res = analyzer.analyze("Noisy text with special characters: #$%^&* 1234567890!")
    assert res.label == "positive"
    assert round(sum(res.probabilities.values()), 2) == 1.0
    assert res.score > 0.0


def test_sentiment_analyzer_long_text():
    analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=True)
    long_text = "Corporate expansion and resilient growth. " * 100
    res = analyzer.analyze(long_text)
    assert -1.0 <= res.score <= 1.0
    if not is_torch_available():
        assert res.label == "positive"
        assert res.score > 0.0
        assert res.inference_source == "offline_fallback_baseline"
    else:
        assert res.inference_source == "pretrained_transformers"
        assert res.is_mock is False
        assert res.label in ("positive", "neutral")
