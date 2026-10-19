"""Unit tests for the FastAPI application endpoints."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from finrisk.api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "FinRisk" in data["service"]


def test_api_portfolio_endpoint():
    response = client.get("/portfolio")
    assert response.status_code == 200
    data = response.json()
    assert "total_market_value" in data
    assert data["total_market_value"] == 17000000.0
    assert data["asset_count"] == 10
    assert len(data["assets"]) == 10


def test_api_scenarios_endpoint():
    response = client.get("/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert "scenarios" in data
    assert data["total_scenarios"] >= 4
    assert "geopolitical_escalation" in data["scenarios"]
    assert "macroeconomic_stagflation" in data["scenarios"]


def test_api_risk_signals_endpoint():
    response = client.get("/risk-signals")
    assert response.status_code == 200
    data = response.json()
    assert "signals" in data
    assert data["total_count"] >= 1

    # Filter by company
    resp_filtered = client.get("/risk-signals?company=AlphaCorp")
    assert resp_filtered.status_code == 200
    filtered_data = resp_filtered.json()
    for sig in filtered_data["signals"]:
        assert "AlphaCorp" in sig["company"]


def test_api_stress_test_preconfigured_scenario():
    payload = {"scenario_id": "geopolitical_escalation"}
    response = client.post("/stress-test", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scenario_id"] == "geopolitical_escalation"
    assert data["baseline_portfolio_value"] == 17000000.0
    assert data["total_absolute_loss"] > 0
    assert data["total_percentage_loss"] > 0
    assert len(data["asset_results"]) == 10
    assert "disclaimer" in data


def test_api_stress_test_custom_parameters():
    payload = {
        "custom_parameters": {
            "equity_shock_pct": -20.0,
            "interest_rate_shock_bps": 100.0,
            "credit_spread_shock_bps": 150.0,
            "loan_loss_shock_pct": 25.0,
        },
        "custom_name": "Hypothetical Custom Shock",
    }
    response = client.post("/stress-test", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scenario_id"] == "custom_scenario"
    assert data["scenario_name"] == "Hypothetical Custom Shock"
    assert data["total_absolute_loss"] > 0


def test_api_stress_test_unknown_scenario_returns_404():
    payload = {"scenario_id": "unknown_fictional_scenario"}
    response = client.post("/stress-test", json=payload)
    assert response.status_code == 404
    assert "Unknown scenario ID" in response.json()["detail"]


def test_api_stress_test_empty_payload_returns_400():
    response = client.post("/stress-test", json={})
    assert response.status_code == 400
    assert "Must provide either 'scenario_id' or 'custom_parameters'" in response.json()["detail"]


def test_api_public_sources_endpoint():
    response = client.get("/public-news/sources")
    assert response.status_code == 200
    data = response.json()
    assert data["total_sources"] >= 2
    feed_ids = [s["feed_id"] for s in data["sources"]]
    assert "federal_reserve" in feed_ids
    assert "sec_press_releases" in feed_ids
    assert "ecb_press_releases" in feed_ids


def test_api_risk_signals_provenance_filtering():
    # Synthetic filter
    resp_synth = client.get("/risk-signals?provenance=synthetic")
    assert resp_synth.status_code == 200
    synth_data = resp_synth.json()
    assert synth_data["provenance"] == "synthetic"
    for s in synth_data["signals"]:
        assert s.get("provenance", "synthetic") == "synthetic"

    # All filter
    resp_all = client.get("/risk-signals?provenance=all")
    assert resp_all.status_code == 200
    assert resp_all.json()["provenance"] == "all"

    # Real public news filter
    resp_real = client.get("/risk-signals?provenance=real_public_news")
    assert resp_real.status_code == 200
    assert resp_real.json()["provenance"] == "real_public_news"


def test_api_fetch_public_news_mocked():
    from unittest.mock import MagicMock, patch

    mock_result = MagicMock()
    mock_result.accepted_records = []
    mock_result.output_count = 0
    mock_result.feed_statuses = []

    with patch("finrisk.ingestion.public_news.fetch_public_news", return_value=mock_result):
        response = client.post("/public-news/fetch")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["articles_fetched"] == 0
        assert data["signals_generated"] == 0


def test_api_suggested_scenarios_endpoint():
    response = client.get("/scenarios/suggested?provenance=all")
    assert response.status_code == 200
    data = response.json()
    assert "total_signals_analyzed" in data
    assert "mapped_signal_count" in data
    assert "unmapped_signal_count" in data
    assert "recommendations" in data
    assert "disclaimer" in data
    assert "Decision Support Only" in data["disclaimer"]


def test_api_match_signals_endpoint():
    signals_payload = [
        {
            "record_id": "test-geo-1",
            "event_type": "Geopolitical",
            "sentiment_score": -0.65,
            "impact_score": 8,
            "text": "Stricter export sanctions placed on advanced technology.",
        },
        {
            "record_id": "test-crd-1",
            "event_type": "Credit Event",
            "sentiment_score": -0.80,
            "impact_score": 9,
            "text": "Commercial real estate bond default triggers credit spread widening.",
        },
    ]
    response = client.post("/scenarios/match-signals", json=signals_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_signals_analyzed"] == 2
    assert data["mapped_signal_count"] == 2
    assert len(data["recommendations"]) == 2
    rec_ids = [r["scenario_id"] for r in data["recommendations"]]
    assert "geopolitical_escalation" in rec_ids
    assert "credit_crunch" in rec_ids


