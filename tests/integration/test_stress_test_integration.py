"""Integration tests for the complete risk-signals-to-portfolio-stress-test workflow."""

from __future__ import annotations

from pathlib import Path
import pytest

from finrisk.stress_testing.portfolio import load_portfolio_csv
from finrisk.stress_testing.scenario_engine import ScenarioEngine, load_scenarios_from_yaml


def test_complete_risk_to_stress_test_integration():
    project_root = Path(__file__).resolve().parent.parent.parent
    portfolio_file = project_root / "data" / "sample" / "sample_portfolio.csv"
    scenarios_file = project_root / "configs" / "stress_scenarios.yaml"
    risk_signals_file = project_root / "data" / "processed" / "risk_signals.csv"

    assert portfolio_file.exists(), f"Missing portfolio file: {portfolio_file}"
    assert scenarios_file.exists(), f"Missing scenarios file: {scenarios_file}"
    assert risk_signals_file.exists(), f"Missing risk signals file: {risk_signals_file}"

    portfolio = load_portfolio_csv(portfolio_file)
    engine = ScenarioEngine.from_config_file(scenarios_file)
    scenarios = load_scenarios_from_yaml(scenarios_file)

    assert len(scenarios) >= 4

    # Run stress test across all preconfigured scenarios
    for sc_id in scenarios.keys():
        result = engine.run_stress_test(portfolio=portfolio, scenario_id=sc_id)

        assert result.baseline_portfolio_value == 17000000.0
        assert result.stressed_portfolio_value > 0.0
        assert result.total_absolute_loss >= 0.0
        assert result.total_percentage_loss >= 0.0
        assert len(result.asset_results) == portfolio.asset_count

        # Total loss must equal sum of asset absolute losses
        asset_sum = sum(a.absolute_loss for a in result.asset_results)
        assert abs(result.total_absolute_loss - asset_sum) < 0.05

        # Check contribution percentages
        if result.total_absolute_loss > 0:
            contrib_sum = sum(a.loss_contribution_pct for a in result.asset_results)
            assert abs(contrib_sum - 100.0) < 0.1


def test_news_events_to_scenario_recommendation_and_execution_pipeline():
    """Verify end-to-end flow from news risk signals to scenario recommendation and execution."""
    import csv
    from finrisk.stress_testing.scenario_mapper import recommend_scenarios_from_signals

    project_root = Path(__file__).resolve().parent.parent.parent
    portfolio_file = project_root / "data" / "sample" / "sample_portfolio.csv"
    scenarios_file = project_root / "configs" / "stress_scenarios.yaml"
    signals_file = project_root / "data" / "processed" / "risk_signals.csv"

    signals = []
    with open(signals_file, mode="r", encoding="utf-8") as f:
        signals = list(csv.DictReader(f))

    assert len(signals) >= 1

    # Step 1: Decision-Support Mapping
    summary = recommend_scenarios_from_signals(signals)
    assert summary.total_signals_analyzed == len(signals)
    assert summary.unique_signals_analyzed <= len(signals)
    assert summary.top_recommendation is not None
    assert summary.top_recommendation.scenario_id in ("geopolitical_escalation", "macroeconomic_stagflation", "credit_crunch", "mild_recession")

    # Step 2: User Confirms and Executes Recommended Scenario
    chosen_scenario_id = summary.top_recommendation.scenario_id
    portfolio = load_portfolio_csv(portfolio_file)
    engine = ScenarioEngine.from_config_file(scenarios_file)

    test_result = engine.run_stress_test(portfolio=portfolio, scenario_id=chosen_scenario_id)
    assert test_result.scenario_id == chosen_scenario_id
    assert test_result.total_absolute_loss > 0.0
    assert test_result.stressed_portfolio_value < test_result.baseline_portfolio_value

    # Mathematical consistency
    asset_losses_sum = sum(a.absolute_loss for a in test_result.asset_results)
    assert abs(test_result.total_absolute_loss - asset_losses_sum) < 0.05

