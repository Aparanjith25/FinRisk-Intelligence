"""Unit tests for the High-Impact Alert detection and recommendation module."""

from __future__ import annotations

import pytest

from finrisk.risk_engine.alerting import (
    DEFAULT_HIGH_IMPACT_THRESHOLD,
    HighImpactAlert,
    detect_high_impact_alerts,
)


def test_default_threshold_boundary():
    """Verify that impact_score must be strictly greater than threshold (> 7 by default)."""
    signals = [
        {"record_id": "sig-6", "impact_score": 6, "event_type": "Macroeconomic", "sentiment_score": -0.6},
        {"record_id": "sig-7", "impact_score": 7, "event_type": "Geopolitical", "sentiment_score": -0.7},
        {"record_id": "sig-8", "impact_score": 8, "event_type": "Credit Event", "sentiment_score": -0.8},
        {"record_id": "sig-10", "impact_score": 10, "event_type": "Macroeconomic", "sentiment_score": -0.9},
    ]

    alerts = detect_high_impact_alerts(signals, threshold=DEFAULT_HIGH_IMPACT_THRESHOLD)

    # Only impact > 7 (i.e. 8 and 10) must trigger
    alert_ids = [a.record_id for a in alerts]
    assert "sig-6" not in alert_ids
    assert "sig-7" not in alert_ids  # Boundary check: 7 is NOT > 7
    assert "sig-8" in alert_ids
    assert "sig-10" in alert_ids
    assert len(alerts) == 2


def test_custom_threshold_configuration():
    """Verify threshold parameter can be configured to other values."""
    signals = [
        {"record_id": "sig-5", "impact_score": 5, "event_type": "Geopolitical"},
        {"record_id": "sig-6", "impact_score": 6, "event_type": "Geopolitical"},
    ]

    # Threshold = 5
    alerts_5 = detect_high_impact_alerts(signals, threshold=5)
    assert len(alerts_5) == 1
    assert alerts_5[0].record_id == "sig-6"

    # Threshold = 8
    alerts_8 = detect_high_impact_alerts(signals, threshold=8)
    assert len(alerts_8) == 0


def test_irrelevant_and_unmapped_event_categories():
    """Verify unmapped or operational events (e.g., M&A, Product Launch) are handled safely."""
    signals = [
        {
            "record_id": "sig-ma",
            "impact_score": 9,
            "event_type": "M&A",
            "sentiment_score": -0.4,
            "text": "Megacorp acquires Competitor for $50B.",
        },
        {
            "record_id": "sig-prod",
            "impact_score": 8,
            "event_type": "Product Launch",
            "sentiment_score": 0.8,
            "text": "Flagship product launched across 30 countries.",
        },
        {
            "record_id": "sig-geo",
            "impact_score": 9,
            "event_type": "Geopolitical",
            "sentiment_score": -0.8,
            "text": "Strict trade embargo announced on energy commodities.",
        },
    ]

    alerts = detect_high_impact_alerts(signals, threshold=7)
    assert len(alerts) == 3

    # M&A alert exists due to high impact, but indicates no macro stress scenario applies
    ma_alert = next(a for a in alerts if a.record_id == "sig-ma")
    assert ma_alert.is_stress_test_applicable is False
    assert ma_alert.recommended_scenario_id is None
    assert "No Preconfigured Stress Scenario" in ma_alert.recommended_scenario_name

    # Geopolitical alert maps to geopolitical_escalation
    geo_alert = next(a for a in alerts if a.record_id == "sig-geo")
    assert geo_alert.is_stress_test_applicable is True
    assert geo_alert.recommended_scenario_id == "geopolitical_escalation"
    assert geo_alert.requires_human_confirmation is True


def test_missing_and_malformed_values():
    """Verify graceful handling of None, empty strings, malformed types, and missing fields."""
    signals = [
        None,
        "not-a-dict",
        {},
        {"record_id": "sig-none", "impact_score": None},
        {"record_id": "sig-empty", "impact_score": ""},
        {"record_id": "sig-str-invalid", "impact_score": "high"},
        {
            "record_id": "sig-str-valid",
            "impact_score": "9",
            "source": None,
            "published_at": None,
            "text": None,
            "sentiment_score": "bad-float",
        },
    ]

    alerts = detect_high_impact_alerts(signals, threshold=7)
    assert len(alerts) == 1
    a = alerts[0]
    assert a.record_id == "sig-str-valid"
    assert a.impact_score == 9
    assert a.source == "Unknown"
    assert a.published_at == "Unknown"
    assert a.summary_text == ""
    assert a.sentiment_score == 0.0


def test_duplicate_signals_deduplication():
    """Verify that duplicate signals by record_id or content signature are deduplicated."""
    signals = [
        {
            "record_id": "sig-dup",
            "impact_score": 9,
            "event_type": "Credit Event",
            "sentiment_score": -0.85,
            "text": "Major bank files for bankruptcy protection.",
        },
        {
            "record_id": "sig-dup",  # Duplicate ID
            "impact_score": 9,
            "event_type": "Credit Event",
            "sentiment_score": -0.85,
            "text": "Major bank files for bankruptcy protection.",
        },
        {
            "record_id": "sig-unique",
            "impact_score": 9,
            "event_type": "Macroeconomic",
            "sentiment_score": -0.70,
            "text": "Central bank hikes benchmark rates by 100 basis points.",
        },
    ]

    # With deduplicate=True (default)
    alerts_dedup = detect_high_impact_alerts(signals, threshold=7, deduplicate=True)
    assert len(alerts_dedup) == 2
    assert [a.record_id for a in alerts_dedup] == ["sig-dup", "sig-unique"]

    # With deduplicate=False
    alerts_all = detect_high_impact_alerts(signals, threshold=7, deduplicate=False)
    assert len(alerts_all) == 3


def test_human_confirmation_requirement_preserved():
    """Verify that all generated alerts explicitly require human confirmation before execution."""
    signals = [
        {
            "record_id": "sig-1",
            "impact_score": 10,
            "event_type": "Geopolitical",
            "sentiment_score": -0.9,
            "text": "War declared.",
        }
    ]
    alerts = detect_high_impact_alerts(signals, threshold=7)
    assert len(alerts) == 1
    assert alerts[0].requires_human_confirmation is True
    dict_rep = alerts[0].to_dict()
    assert dict_rep["requires_human_confirmation"] is True
    assert dict_rep["impact_score"] == 10
