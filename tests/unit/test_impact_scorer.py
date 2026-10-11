"""Unit tests for the financial impact scorer module."""

from __future__ import annotations

import pytest

from finrisk.risk_engine.impact_scorer import compute_impact_score


def test_impact_score_integer_bounds():
    # Maximum extreme combination
    max_res = compute_impact_score(
        event_category="Credit Event",
        sentiment_score=-0.95,
        sentiment_confidence=0.95,
        company="GlobalMegaCorp",
        text="Debt default exceeded $500 million in bondholder obligations.",
    )
    assert isinstance(max_res.score, int)
    assert 1 <= max_res.score <= 10
    assert max_res.score >= 9  # Should be near top of scale

    # Minimum extreme combination
    min_res = compute_impact_score(
        event_category="Other",
        sentiment_score=0.0,
        sentiment_confidence=0.20,  # Penalty for low confidence
        company=None,
        text="General routine observation with no metrics.",
    )
    assert isinstance(min_res.score, int)
    assert 1 <= min_res.score <= 10
    assert min_res.score <= 2  # Should be at the bottom of the scale


def test_impact_score_missing_and_none_inputs():
    # All optional fields None
    res = compute_impact_score(
        event_category=None,
        sentiment_score=None,
        sentiment_confidence=None,
        company=None,
        text=None,
    )
    assert isinstance(res.score, int)
    assert 1 <= res.score <= 10
    assert res.breakdown["event_category"] == "Other"
    assert res.breakdown["sentiment_adjustment"] == 0
    assert res.breakdown["entity_adjustment"] == 0
    assert res.breakdown["evidence_adjustment"] == 0


def test_impact_score_quantitative_evidence():
    with_numbers = compute_impact_score(
        event_category="Macroeconomic",
        sentiment_score=-0.4,
        company="FederalReserve",
        text="Inflation rose by 4.2 percent and yields jumped 25 bps.",
    )
    without_numbers = compute_impact_score(
        event_category="Macroeconomic",
        sentiment_score=-0.4,
        company="FederalReserve",
        text="Inflation rose higher and yields climbed across curves.",
    )
    # The one with quantitative numbers gets +1 evidence adjustment
    assert with_numbers.score == without_numbers.score + 1
    assert with_numbers.breakdown["evidence_adjustment"] == 1
    assert without_numbers.breakdown["evidence_adjustment"] == 0


def test_impact_score_limitations_and_rationale():
    res = compute_impact_score(
        event_category="Geopolitical",
        sentiment_score=-0.5,
        company="EnergyCorp",
        text="Crude pipeline suspended under sanctions.",
    )
    assert "not a calibrated" in res.limitations.lower()
    assert "Base score" in res.rationale
    assert "Final impact:" in res.rationale
