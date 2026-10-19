"""Test suite for SourceAwareSentimentRouter prototype.

Tests:
1. Regulatory routing paths (Fed, SEC, ECB).
2. Corporate routing paths (PhraseBank, earnings, company press).
3. Fallback routing on unknown, empty, or ambiguous sources.
4. Edge cases (empty string, whitespace, long text).
5. Exact formula verification (score == P(pos) - P(neg)).
6. Downstream risk engine compatibility.
"""

from __future__ import annotations

import os
import sys
import pytest

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.finrisk_model_improvements.routing_prototype.router import (
    SourceAwareSentimentRouter,
    RoutedSentimentResult,
)
from finrisk.nlp.sentiment_analyzer import SentimentResult
from finrisk.stress_testing.scenario_mapper import map_signal_to_scenario


@pytest.fixture(scope="module")
def router():
    return SourceAwareSentimentRouter()


def test_regulatory_routing_paths(router):
    regulatory_sources = [
        "U.S. Federal Reserve Press Releases",
        "federal_reserve",
        "U.S. SEC Press Releases",
        "sec_press_releases",
        "sec_edgar",
        "European Central Bank Press Releases",
        "ecb_press_releases",
    ]
    for src in regulatory_sources:
        choice, cat = router.route_source(src)
        assert choice == "regulatory_model", f"Failed for source '{src}': got {choice}"
        assert cat == "regulatory_official"

        res = router.analyze("Federal Reserve issues enforcement action.", source=src)
        assert res.routed_engine == "regulatory_model"
        assert res.label in {"positive", "negative", "neutral"}
        assert -1.0 <= res.score <= 1.0


def test_corporate_routing_paths(router):
    corporate_sources = [
        "financial_phrasebank",
        "corporate_news",
        "company_press",
        "earnings_release",
        "pr_newswire",
    ]
    for src in corporate_sources:
        choice, cat = router.route_source(src)
        assert choice == "corporate_model", f"Failed for source '{src}': got {choice}"
        assert cat == "corporate_financial"

        res = router.analyze("Operating profit surged 25% year over year.", source=src)
        assert res.routed_engine == "corporate_model"
        assert res.label in {"positive", "negative", "neutral"}
        assert -1.0 <= res.score <= 1.0


def test_unknown_and_ambiguous_source_fallback(router):
    unknown_sources = [
        None,
        "",
        "   ",
        "untrusted_web_scraper",
        "synthetic_data",
        "random_market_blog",
        "crypto_telegram_channel",
    ]
    for src in unknown_sources:
        choice, cat = router.route_source(src)
        assert choice == "fallback_model", f"Failed fallback for '{src}': got {choice}"
        assert "fallback" in cat or "empty" in cat

        res = router.analyze("General financial market sentiment overview.", source=src)
        assert res.routed_engine == "fallback_model"
        assert "prosus_finbert_baseline" in res.model_name


def test_score_formula_and_bounds(router):
    test_text = "Net income fell sharply due to credit loss provisions."
    res = router.analyze(test_text, source="corporate_news")

    p_pos = res.probabilities["positive"]
    p_neg = res.probabilities["negative"]
    p_neu = res.probabilities["neutral"]

    expected_score = round(p_pos - p_neg, 4)
    assert abs(res.score - expected_score) < 1e-4, f"Score mismatch: {res.score} vs {expected_score}"
    assert -1.0 <= res.score <= 1.0
    assert abs((p_pos + p_neg + p_neu) - 1.0) < 1e-3


def test_edge_cases_handling(router):
    # Empty string
    res_empty = router.analyze("", source="corporate_news")
    assert res_empty.label == "neutral"
    assert res_empty.score == 0.0

    # Whitespace
    res_ws = router.analyze("    \n\t   ", source="sec_press_releases")
    assert res_ws.label == "neutral"
    assert res_ws.score == 0.0

    # Very long text
    long_text = "Financial risk assessment and bank credit portfolio exposure. " * 300
    res_long = router.analyze(long_text, source="federal_reserve")
    assert res_long.label in {"positive", "negative", "neutral"}
    assert -1.0 <= res_long.score <= 1.0


def test_downstream_scenario_mapper_compatibility(router):
    # Verify that RoutedSentimentResult works seamlessly with map_signal_to_scenario
    res = router.analyze("SEC announces civil fraud enforcement action.", source="sec_press_releases")

    signal_dict = {
        "record_id": "routed_sig_001",
        "event_type": "Credit Event",
        "sentiment_score": res.score,
        "impact_score": 8,
        "text": "SEC announces civil fraud enforcement action.",
        "source": "U.S. SEC Press Releases",
    }
    match_res = map_signal_to_scenario(signal_dict)
    assert match_res.scenario_id == "credit_crunch"
    assert match_res.is_mapped is True
