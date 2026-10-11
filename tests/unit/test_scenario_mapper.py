"""Unit tests for news-to-scenario stress testing mapping."""

from __future__ import annotations

import pytest

from finrisk.stress_testing.scenario_mapper import (
    AggregatedScenarioRecommendation,
    ScenarioMappingSummary,
    ScenarioMatch,
    map_signal_to_scenario,
    recommend_scenarios_from_signals,
)


def test_map_geopolitical_signal():
    signal = {
        "record_id": "geo-1",
        "event_type": "Geopolitical",
        "sentiment_score": -0.65,
        "impact_score": 7,
        "text": "Escalating trade war tariffs and export restrictions imposed on key minerals.",
    }
    match = map_signal_to_scenario(signal)
    assert match.is_mapped is True
    assert match.scenario_id == "geopolitical_escalation"
    assert match.category == "Geopolitical"
    assert match.relevance_score >= 0.70
    assert "Geopolitical Escalation" in match.scenario_name
    assert "Geopolitical risk event" in match.rationale


def test_map_macroeconomic_stagflation_signal():
    signal = {
        "record_id": "macro-1",
        "event_type": "Macroeconomic",
        "sentiment_score": -0.50,
        "impact_score": 8,
        "text": "Stubborn CPI inflation forced central bank rate hikes, sparking stagflation fears.",
    }
    match = map_signal_to_scenario(signal)
    assert match.is_mapped is True
    assert match.scenario_id == "macroeconomic_stagflation"
    assert match.category == "Macroeconomic"
    assert match.relevance_score >= 0.80


def test_map_macroeconomic_mild_recession_signal():
    signal = {
        "record_id": "macro-2",
        "event_type": "Macroeconomic",
        "sentiment_score": -0.20,
        "impact_score": 4,
        "text": "Modest economic slowdown with minor uptick in unemployment rate and cooling demand.",
    }
    match = map_signal_to_scenario(signal)
    assert match.is_mapped is True
    assert match.scenario_id == "mild_recession"
    assert match.category == "Macroeconomic"
    assert match.relevance_score > 0.50


def test_map_credit_crunch_signal():
    signal = {
        "record_id": "crd-1",
        "event_type": "Credit Event",
        "sentiment_score": -0.80,
        "impact_score": 9,
        "text": "Surge in high-yield debt defaults and massive rating downgrades across real estate sector.",
    }
    match = map_signal_to_scenario(signal)
    assert match.is_mapped is True
    assert match.scenario_id == "credit_crunch"
    assert match.category == "Credit Event"
    assert match.relevance_score >= 0.90


def test_unmapped_operational_and_microeconomic_events():
    # 1. Merger / Acquisition
    ma_sig = {
        "record_id": "ma-1",
        "event_type": "Merger/Acquisition",
        "sentiment_score": 0.40,
        "impact_score": 6,
        "text": "Industrial company completed strategic acquisition of regional parts distributor.",
    }
    ma_match = map_signal_to_scenario(ma_sig)
    assert ma_match.is_mapped is False
    assert ma_match.scenario_id is None
    assert "Microeconomic event" in ma_match.rationale
    assert "M&A" in ma_match.unmapped_reason

    # 2. Product Launch
    prod_sig = {
        "record_id": "prod-1",
        "event_type": "Product Launch",
        "sentiment_score": 0.70,
        "impact_score": 5,
        "text": "Automaker unveiled next-generation electric vehicle platform.",
    }
    prod_match = map_signal_to_scenario(prod_sig)
    assert prod_match.is_mapped is False
    assert prod_match.scenario_id is None
    assert "Product Launch" in prod_match.unmapped_reason

    # 3. Other (Routine Corporate)
    other_sig = {
        "record_id": "oth-1",
        "event_type": "Other",
        "sentiment_score": 0.05,
        "impact_score": 2,
        "text": "Annual general meeting held in conference center without resolution controversies.",
    }
    oth_match = map_signal_to_scenario(other_sig)
    assert oth_match.is_mapped is False
    assert oth_match.scenario_id is None


def test_recommend_scenarios_deduplication():
    # Duplicate signals with identical record_id should be counted only once
    dup_sig = {
        "record_id": "geo-dup-1",
        "event_type": "Geopolitical",
        "sentiment_score": -0.70,
        "impact_score": 8,
        "text": "Heavy trade sanctions placed on metals.",
    }
    signals = [dup_sig, dup_sig, dup_sig]

    summary = recommend_scenarios_from_signals(signals)
    assert summary.total_signals_analyzed == 3
    assert summary.unique_signals_analyzed == 1
    assert summary.mapped_signal_count == 1
    assert summary.top_recommendation is not None
    assert summary.top_recommendation.trigger_count == 1
    assert summary.top_recommendation.scenario_id == "geopolitical_escalation"


def test_recommend_scenarios_aggregation_and_ranking():
    signals = [
        # 3 Geopolitical signals
        {"record_id": "g1", "event_type": "Geopolitical", "sentiment_score": -0.6, "impact_score": 8, "text": "Sanctions"},
        {"record_id": "g2", "event_type": "Geopolitical", "sentiment_score": -0.7, "impact_score": 7, "text": "Tariffs"},
        {"record_id": "g3", "event_type": "Geopolitical", "sentiment_score": -0.5, "impact_score": 9, "text": "War tensions"},
        # 1 Credit signal
        {"record_id": "c1", "event_type": "Credit Event", "sentiment_score": -0.8, "impact_score": 8, "text": "Defaults"},
        # 2 Unmapped operational signals
        {"record_id": "m1", "event_type": "Merger/Acquisition", "sentiment_score": 0.3, "impact_score": 5, "text": "Acquisition"},
        {"record_id": "p1", "event_type": "Product Launch", "sentiment_score": 0.6, "impact_score": 4, "text": "New model"},
    ]

    summary = recommend_scenarios_from_signals(signals)
    assert summary.total_signals_analyzed == 6
    assert summary.unique_signals_analyzed == 6
    assert summary.mapped_signal_count == 4
    assert summary.unmapped_signal_count == 2
    assert len(summary.unmapped_breakdown) >= 2

    # Geopolitical should rank #1 because it has 3 triggers vs 1 credit trigger
    assert summary.top_recommendation is not None
    assert summary.top_recommendation.scenario_id == "geopolitical_escalation"
    assert summary.top_recommendation.trigger_count == 3
    assert summary.top_recommendation.average_impact == 8.0
    assert len(summary.top_recommendation.contributing_signal_ids) == 3

    # Credit crunch should be second recommendation
    rec_ids = [r.scenario_id for r in summary.recommendations]
    assert "geopolitical_escalation" in rec_ids
    assert "credit_crunch" in rec_ids

    # Dictionary export check
    summary_dict = summary.to_dict()
    assert isinstance(summary_dict, dict)
    assert summary_dict["top_recommendation"]["scenario_id"] == "geopolitical_escalation"
    assert "Decision Support Only" in summary_dict["disclaimer"]


def test_recommend_scenarios_empty_list():
    summary = recommend_scenarios_from_signals([])
    assert summary.total_signals_analyzed == 0
    assert summary.mapped_signal_count == 0
    assert summary.top_recommendation is None
    assert summary.recommendations == []


def test_scenario_mapper_edge_cases():
    # 1. Non-dict input
    res_none = map_signal_to_scenario(None)  # type: ignore
    assert res_none.is_mapped is False
    assert res_none.scenario_id is None
    assert "Malformed" in res_none.rationale

    # 2. Extreme and invalid score types
    bad_score_sig = {
        "record_id": "bad-1",
        "event_type": "Geopolitical",
        "sentiment_score": "not_a_float",
        "impact_score": "not_an_int",
        "text": "Trade negotiations ongoing.",
    }
    res_bad = map_signal_to_scenario(bad_score_sig)
    # Impact score defaults to 0 -> unmapped (< 3)
    assert res_bad.is_mapped is False

    # 3. Clamping of extreme scores
    extreme_sig = {
        "record_id": "ext-1",
        "event_type": "Geopolitical",
        "sentiment_score": -99.0,  # Clamped to -1.0
        "impact_score": 999,       # Clamped to 10
        "text": "Heavy military escalation and export bans.",
    }
    res_ext = map_signal_to_scenario(extreme_sig)
    assert res_ext.is_mapped is True
    assert res_ext.scenario_id == "geopolitical_escalation"
    assert res_ext.relevance_score <= 1.0

    # 4. Deduping signals with missing record_id by content hash
    unid_sig1 = {"event_type": "Geopolitical", "sentiment_score": -0.8, "impact_score": 7, "text": "Same news text"}
    unid_sig2 = {"event_type": "Geopolitical", "sentiment_score": -0.8, "impact_score": 7, "text": "Same news text"}
    summary = recommend_scenarios_from_signals([unid_sig1, unid_sig2, "invalid_string"])  # type: ignore
    assert summary.total_signals_analyzed == 3
    assert summary.unique_signals_analyzed == 1
    assert summary.mapped_signal_count == 1

