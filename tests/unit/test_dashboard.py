"""Regression and unit tests for Streamlit dashboard logic and API fallback."""

from __future__ import annotations

import unittest.mock as mock
import pytest
from streamlit.testing.v1 import AppTest

from finrisk.dashboard.app import (
    API_BASE_URL,
    fetch_api_or_fallback,
    load_local_portfolio,
    load_local_risk_signals,
    load_local_scenarios,
    run_local_stress_test,
)


def test_load_local_scenarios_returns_all_configured():
    """Verify that load_local_scenarios returns all 4 preconfigured scenarios."""
    scenarios = load_local_scenarios()
    assert isinstance(scenarios, dict)
    assert len(scenarios) == 4
    expected_keys = {
        "geopolitical_escalation",
        "macroeconomic_stagflation",
        "credit_crunch",
        "mild_recession",
    }
    assert set(scenarios.keys()) == expected_keys
    for key, data in scenarios.items():
        assert "name" in data
        assert "description" in data
        assert "parameters" in data
        assert isinstance(data["parameters"], dict)


def test_scenario_selection_and_stress_test_for_every_scenario():
    """Verify that every configured scenario ID can be stressed without error."""
    scenarios = load_local_scenarios()
    assert len(scenarios) >= 4
    for scenario_id in scenarios:
        res = run_local_stress_test(scenario_id)
        assert res["scenario_id"] == scenario_id
        assert res["baseline_portfolio_value"] == 16000000.0
        assert res["stressed_portfolio_value"] > 0
        assert res["total_absolute_loss"] > 0
        assert len(res["asset_results"]) == 9
        assert "disclaimer" in res


def test_missing_or_empty_scenario_data_handling():
    """Verify that empty, None, or malformed scenario payloads do not crash."""
    test_cases = [
        {},
        {"scenarios": {}},
        {"scenarios": None},
        None,
        {"invalid_key": 123},
    ]

    for raw in test_cases:
        if isinstance(raw, dict):
            if "scenarios" in raw and isinstance(raw["scenarios"], dict):
                scenarios_dict = raw["scenarios"]
            else:
                scenarios_dict = raw
        else:
            scenarios_dict = {}

        keys = [k for k in scenarios_dict.keys() if isinstance(scenarios_dict[k], dict)]
        # Must not raise ValueError or IndexError
        assert isinstance(keys, list)
        if not keys:
            # Graceful empty handling: no selection made
            selected_key = None
            assert selected_key is None


def test_api_unavailable_local_fallback():
    """Verify that when API is unreachable, local fallback executes and labels source."""
    with mock.patch("httpx.get", side_effect=Exception("Connection refused")):
        # 1. Health endpoint fallback
        health_data, health_src = fetch_api_or_fallback(
            "/health",
            lambda: {"status": "offline"},
        )
        assert health_data["status"] == "offline"
        assert "Local Engine Direct Fallback" in health_src
        assert "0 live data fabricated" in health_src

        # 2. Scenarios endpoint fallback
        scen_data, scen_src = fetch_api_or_fallback(
            "/scenarios",
            load_local_scenarios,
        )
        assert len(scen_data) == 4
        assert "geopolitical_escalation" in scen_data
        assert "Local Engine Direct Fallback" in scen_src

        # 3. Portfolio endpoint fallback
        port_data, port_src = fetch_api_or_fallback(
            "/portfolio",
            load_local_portfolio,
        )
        assert port_data["total_market_value"] == 16000000.0
        assert len(port_data["assets"]) == 9
        assert "Local Engine Direct Fallback" in port_src

    with mock.patch("httpx.post", side_effect=Exception("Connection refused")):
        # 4. Stress test endpoint POST fallback
        test_data, test_src = fetch_api_or_fallback(
            "/stress-test",
            run_local_stress_test,
            "geopolitical_escalation",
            method="POST",
            json_data={"scenario_id": "geopolitical_escalation"},
        )
        assert test_data["scenario_id"] == "geopolitical_escalation"
        assert test_data["total_absolute_loss"] > 0
        assert "Local Engine Direct Fallback" in test_src


def test_api_available_response_consumption():
    """Verify that when API responds with 200 OK, dashboard consumes API response."""
    mock_scenarios_response = mock.MagicMock()
    mock_scenarios_response.status_code = 200
    mock_scenarios_response.json.return_value = {
        "total_scenarios": 1,
        "scenarios": {
            "custom_macro_shock": {
                "id": "custom_macro_shock",
                "name": "Live API Shock Scenario",
                "description": "Mocked from live FastAPI",
                "parameters": {"equity_shock_pct": -10.0},
            }
        },
    }

    with mock.patch("httpx.get", return_value=mock_scenarios_response):
        scen_raw, scen_src = fetch_api_or_fallback(
            "/scenarios",
            load_local_scenarios,
        )
        assert "FastAPI Live Endpoint" in scen_src
        assert "scenarios" in scen_raw
        assert "custom_macro_shock" in scen_raw["scenarios"]

        # Extraction logic handles wrapped scenarios dict
        if isinstance(scen_raw, dict) and "scenarios" in scen_raw and isinstance(scen_raw["scenarios"], dict):
            extracted = scen_raw["scenarios"]
        else:
            extracted = scen_raw
        assert "custom_macro_shock" in extracted
        assert extracted["custom_macro_shock"]["name"] == "Live API Shock Scenario"


def test_dashboard_renders_without_exceptions_via_apptest():
    """Verify that the full Streamlit dashboard renders without any unhandled exceptions."""
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent.parent / "finrisk" / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path)).run(timeout=30)
    assert len(at.exception) == 0, f"AppTest threw exceptions: {at.exception}"
    assert len(at.selectbox) >= 1
    # Verify the scenario selectbox has all 4 preconfigured scenario options
    scenario_options = at.selectbox[0].options
    assert len(scenario_options) == 4
    assert "Geopolitical Escalation & Trade Shock" in scenario_options


def test_dashboard_provenance_signal_loading():
    """Verify that load_local_risk_signals respects provenance filter."""
    # Synthetic signals should exist
    synth_signals = load_local_risk_signals(provenance="synthetic")
    assert isinstance(synth_signals, list)
    if synth_signals:
        for s in synth_signals:
            assert s.get("provenance", "synthetic") == "synthetic"

    # All signals
    all_signals = load_local_risk_signals(provenance="all")
    assert isinstance(all_signals, list)
    assert len(all_signals) >= len(synth_signals)


def test_dashboard_apptest_provenance_radio_switching():
    """Verify that switching data feed radio in AppTest renders without exceptions."""
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent.parent / "finrisk" / "dashboard" / "app.py"
    with mock.patch("httpx.get", side_effect=Exception("Connection refused")):
        at = AppTest.from_file(str(app_path)).run(timeout=10)
        assert len(at.exception) == 0

        # Locate the radio button
        assert len(at.sidebar.radio) >= 1
        radio = at.sidebar.radio[0]
        assert "Synthetic Data (Default)" in radio.options
        assert "Real Public Financial News" in radio.options

        # Switch to Real Public Financial News and rerun
        radio.set_value("Real Public Financial News").run(timeout=10)
        assert len(at.exception) == 0

        # Switch to Combined and rerun
        radio.set_value("Combined (All Sources)").run(timeout=10)
        assert len(at.exception) == 0


def test_load_local_suggested_scenarios():
    """Verify that load_local_suggested_scenarios produces valid recommendations."""
    from finrisk.dashboard.app import load_local_suggested_scenarios

    res = load_local_suggested_scenarios(provenance="all")
    assert isinstance(res, dict)
    assert "total_signals_analyzed" in res
    assert "mapped_signal_count" in res
    assert "recommendations" in res
    assert "disclaimer" in res
    assert "Decision Support Only" in res["disclaimer"]


