"""Unit tests for scenario stress testing engine and hand-calculated valuation formulas."""

from __future__ import annotations

from pathlib import Path
import pytest

from finrisk.stress_testing.portfolio import Portfolio, PortfolioAsset
from finrisk.stress_testing.scenario_engine import ScenarioEngine, load_scenarios_from_yaml


def test_hand_calculated_equity_shock():
    # Equity holding: $1,000,000 with -15% equity shock
    asset = PortfolioAsset(
        asset_id="EQ_TEST",
        asset_name="Test Tech Stock",
        asset_type="equity",
        market_value=1000000.0,
    )
    engine = ScenarioEngine()
    params = {"equity_shock_pct": -15.0}

    res = engine.stress_asset(asset, params)
    # Stressed = 1,000,000 * (1 - 0.15) = 850,000
    assert res.stressed_value == 850000.0
    assert res.absolute_loss == 150000.0
    assert res.percentage_loss == 15.0


def test_hand_calculated_government_bond_shock():
    # Government Bond: $1,000,000, Duration = 5.0 years, +100 bps rate shock (0.01)
    asset = PortfolioAsset(
        asset_id="GB_TEST",
        asset_name="Test Sovereign Bond",
        asset_type="government_bond",
        market_value=1000000.0,
        duration=5.0,
    )
    engine = ScenarioEngine()
    params = {"interest_rate_shock_bps": 100.0}

    res = engine.stress_asset(asset, params)
    # ΔP = -5.0 * (100 / 10000) = -0.05
    # Stressed = 1,000,000 * (1 - 0.05) = 950,000
    assert res.stressed_value == 950000.0
    assert res.absolute_loss == 50000.0
    assert res.percentage_loss == 5.0


def test_hand_calculated_corporate_bond_shock():
    # Corporate Bond: $1,000,000, Duration = 4.0 years
    # +100 bps rate shock (0.01) + 150 bps spread widening (0.015) = total yield increase 0.025
    asset = PortfolioAsset(
        asset_id="CB_TEST",
        asset_name="Test Corporate Bond",
        asset_type="corporate_bond",
        market_value=1000000.0,
        duration=4.0,
        spread_bps=150.0,
    )
    engine = ScenarioEngine()
    params = {"interest_rate_shock_bps": 100.0, "credit_spread_shock_bps": 150.0}

    res = engine.stress_asset(asset, params)
    # ΔP = -4.0 * ((100 + 150) / 10000) = -4.0 * 0.025 = -0.10
    # Stressed = 1,000,000 * (1 - 0.10) = 900,000
    assert res.stressed_value == 900000.0
    assert res.absolute_loss == 100000.0
    assert res.percentage_loss == 10.0


def test_hand_calculated_loan_shock():
    # Loan: $1,000,000, baseline PD = 4% (0.04), LGD = 50% (0.50), shock = 20%
    asset = PortfolioAsset(
        asset_id="LN_TEST",
        asset_name="Test Commercial Loan",
        asset_type="loan",
        market_value=1000000.0,
        baseline_pd=0.04,
        lgd=0.50,
    )
    engine = ScenarioEngine()
    params = {"loan_loss_shock_pct": 20.0}

    res = engine.stress_asset(asset, params)
    # loss_rate = 0.04 * 0.50 * (1 + 20/100) = 0.02 * 1.20 = 0.024
    # Stressed = 1,000,000 * (1 - 0.024) = 976,000
    assert res.stressed_value == 976000.0
    assert res.absolute_loss == 24000.0
    assert res.percentage_loss == 2.4


def test_portfolio_consolidation_and_loss_contributions():
    assets = [
        PortfolioAsset("E1", "Stock A", "equity", 1000000.0),
        PortfolioAsset("G1", "Gov Bond A", "government_bond", 1000000.0, duration=5.0),
    ]
    portfolio = Portfolio(assets=assets)
    engine = ScenarioEngine()
    params = {"equity_shock_pct": -10.0, "interest_rate_shock_bps": 100.0}

    res = engine.run_stress_test(portfolio, custom_parameters=params)
    assert res.baseline_portfolio_value == 2000000.0
    # Stock loss = 100,000; Gov bond loss = 50,000; Total = 150,000
    assert res.total_absolute_loss == 150000.0
    assert res.total_percentage_loss == 7.5

    # Asset loss contributions
    contrib_sum = sum(a.loss_contribution_pct for a in res.asset_results)
    assert abs(contrib_sum - 100.0) < 0.01
    assert res.asset_results[0].loss_contribution_pct == 66.67
    assert res.asset_results[1].loss_contribution_pct == 33.33


def test_load_scenarios_from_yaml():
    project_root = Path(__file__).resolve().parent.parent.parent
    config_file = project_root / "configs" / "stress_scenarios.yaml"
    assert config_file.exists()

    scenarios = load_scenarios_from_yaml(config_file)
    assert "geopolitical_escalation" in scenarios
    assert "macroeconomic_stagflation" in scenarios
    assert "credit_crunch" in scenarios
    assert "mild_recession" in scenarios


def test_unknown_scenario_raises():
    portfolio = Portfolio(assets=[PortfolioAsset("E1", "Stock", "equity", 1000.0)])
    engine = ScenarioEngine(scenarios={"valid": {"parameters": {}}})
    with pytest.raises(ValueError, match="Unknown scenario ID"):
        engine.run_stress_test(portfolio, scenario_id="nonexistent_id")
