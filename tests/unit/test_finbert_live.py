"""Unit and integration test for live Pretrained ProsusAI/FinBERT neural inference.

Skips gracefully if PyTorch/Transformers are not installed or if host commit memory is exhausted.
"""

from __future__ import annotations

import os
import pytest

# Enforce offline Hugging Face configuration and isolated temp storage if E:\hf_home exists
if os.path.exists(r"E:\hf_home"):
    os.environ["HF_HOME"] = r"E:\hf_home"
    os.environ["HF_HUB_CACHE"] = r"E:\hf_home\hub"
    os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_home"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if os.path.exists(r"E:\pip_temp"):
    os.environ["TEMP"] = r"E:\pip_temp"
    os.environ["TMP"] = r"E:\pip_temp"

from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer, FINBERT_LABELS


def is_transformers_available() -> bool:
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


_PROBE_CHECK_DONE = False
_PROBE_ERROR_MSG = ""


def check_live_model_availability() -> str:
    """Probe whether pretrained model can load into memory in current environment."""
    global _PROBE_CHECK_DONE, _PROBE_ERROR_MSG
    if _PROBE_CHECK_DONE:
        return _PROBE_ERROR_MSG

    _PROBE_CHECK_DONE = True
    if not is_transformers_available():
        _PROBE_ERROR_MSG = "PyTorch or Transformers not installed in current environment"
        return _PROBE_ERROR_MSG

    try:
        analyzer = FinBERTSentimentAnalyzer(allow_offline_fallback=False)
        analyzer.analyze("Probe query to verify host memory and weights availability.")
        _PROBE_ERROR_MSG = ""
    except Exception as err:
        _PROBE_ERROR_MSG = f"Host memory or weight loading constraint: {err}"

    return _PROBE_ERROR_MSG


SKIP_REASON = check_live_model_availability()


@pytest.mark.skipif(bool(SKIP_REASON), reason=SKIP_REASON or "Live FinBERT unavailable")
class TestLiveFinBERTInference:
    """Live verification of pretrained FinBERT inference."""

    @pytest.fixture(scope="class")
    def analyzer(self):
        return FinBERTSentimentAnalyzer(allow_offline_fallback=False)

    def test_live_finbert_metadata_contract(self, analyzer):
        text = "Federal Reserve raises benchmark interest rates by 50 bps, signaling persistent inflation concerns."
        res = analyzer.analyze(text)

        assert res.is_mock is False, "Expected is_mock == False for live FinBERT"
        assert res.inference_source == "pretrained_transformers", f"Expected pretrained_transformers, got {res.inference_source}"
        assert res.model_name == "ProsusAI/finbert"
        assert res.label in FINBERT_LABELS
        assert 0.0 <= res.confidence <= 1.0
        assert -1.0 <= res.score <= 1.0

        # Check probability distribution sums to 1.0
        prob_sum = sum(res.probabilities.values())
        assert abs(prob_sum - 1.0) < 0.05, f"Probabilities do not sum to 1: {prob_sum}"

    def test_live_finbert_positive_and_negative_distinction(self, analyzer):
        pos_text = "Company announces record quarterly profits, strong balance sheet resilience, and upgraded dividend."
        neg_text = "Company reports catastrophic losses, severe liquidity crisis, and immediate covenant default."

        pos_res = analyzer.analyze(pos_text)
        neg_res = analyzer.analyze(neg_text)

        assert pos_res.label == "positive"
        assert pos_res.score > 0.0
        assert neg_res.label == "negative"
        assert neg_res.score < 0.0

    def test_live_finbert_long_text_token_truncation(self, analyzer):
        # Text exceeding 512 subword tokens
        long_paragraph = (
            "The Federal Open Market Committee decided today to raise the target range for the federal funds rate. "
            "Recent indicators suggest that economic activity has continued to expand at a modest pace. "
            "Job gains have been solid in recent months, and the unemployment rate has remained low. "
            "Inflation has eased over the past year but remains elevated. In support of its goals, the Committee decided. "
        ) * 20  # ~1600 tokens

        res = analyzer.analyze(long_paragraph)
        assert res.inference_source == "pretrained_transformers"
        assert res.label in FINBERT_LABELS
        assert -1.0 <= res.score <= 1.0

    def test_live_finbert_batch_inference(self, analyzer):
        texts = [
            "Operating margins expanded significantly across all operating divisions.",
            "Credit rating agency downgrades senior unsecured debt to junk status.",
            "The administrative committee will convene for its scheduled annual symposium.",
        ]
        results = analyzer.analyze_batch(texts)
        assert len(results) == 3
        for r in results:
            assert r.inference_source == "pretrained_transformers"
            assert r.is_mock is False
