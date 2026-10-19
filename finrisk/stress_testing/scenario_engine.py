"""Scenario stress testing engine.

Evaluates portfolio resilience under macroeconomic, geopolitical, and credit shocks
using transparent, documented, duration- and valuation-based approximations.
NOTE: These calculations are deterministic sensitivity approximations, NOT calibrated
market predictions or econometric forecasts.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from finrisk.stress_testing.portfolio import Portfolio, PortfolioAsset

logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = (
    "Simulated stress test results are deterministic valuation approximations based on "
    "stylized scenario shocks. They do not constitute financial advice, credit ratings, "
    "or calibrated econometric predictions of actual portfolio market losses."
)


@dataclass
class AssetStressResult:
    """Stress test outcome for an individual portfolio holding."""

    asset_id: str
    asset_name: str
    asset_type: str
    baseline_value: float
    stressed_value: float
    absolute_loss: float
    percentage_loss: float
    loss_contribution_pct: float
    applied_shocks: Dict[str, Any]
    valuation_method: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StressTestResult:
    """Consolidated portfolio stress testing outcome."""

    scenario_id: str
    scenario_name: str
    category: str
    baseline_portfolio_value: float
    stressed_portfolio_value: float
    total_absolute_loss: float
    total_percentage_loss: float
    asset_results: List[AssetStressResult]
    scenario_parameters: Dict[str, Any]
    disclaimer: str = DISCLAIMER_TEXT

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["asset_results"] = [a.to_dict() for a in self.asset_results]
        return res


def load_scenarios_from_yaml(config_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Load scenario definitions from a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Scenarios configuration file not found: {path}")

    with open(path, mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "scenarios" not in data:
        raise ValueError(f"Invalid scenarios configuration format in {path}: expected 'scenarios' root key")

    return data["scenarios"]


class ScenarioEngine:
    """Engine executing deterministic valuation shocks across asset classes."""

    def __init__(self, scenarios: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Initialize with predefined or loaded scenarios dictionary."""
        self.scenarios = scenarios or {}

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> ScenarioEngine:
        """Instantiate engine by loading scenarios from YAML config."""
        scenarios = load_scenarios_from_yaml(config_path)
        return cls(scenarios=scenarios)

    def stress_asset(
        self,
        asset: PortfolioAsset,
        parameters: Dict[str, float],
    ) -> AssetStressResult:
        """Apply asset-specific valuation shock formula.

        Formulas:
        1. Equity: Stressed = Baseline * (1 + equity_shock_pct / 100)
        2. Government Bond: Stressed = Baseline * (1 - duration * (rate_shock_bps / 10000))
        3. Corporate Bond: Stressed = Baseline * (1 - duration * ((rate_shock_bps + spread_shock_bps) / 10000))
        4. Loan: Stressed = Baseline * (1 - (baseline_pd * lgd * (1 + loan_loss_shock_pct / 100)))
        5. Derivative (IRS): Stressed = Baseline ± (DV01_$ * rate_shock_bps) (pay-fixed gains on hikes; receive-fixed gains on cuts)
        """
        base_val = asset.market_value
        eq_shock_pct = float(parameters.get("equity_shock_pct", 0.0))
        rate_shock_bps = float(parameters.get("interest_rate_shock_bps", 0.0))
        spread_shock_bps = float(parameters.get("credit_spread_shock_bps", 0.0))
        loan_loss_shock_pct = float(parameters.get("loan_loss_shock_pct", 0.0))

        applied_shocks: Dict[str, Any] = {}
        valuation_method = ""

        if asset.asset_type == "equity":
            # Direct percentage price shock
            ret = eq_shock_pct / 100.0
            stressed_val = max(0.0, base_val * (1.0 + ret))
            applied_shocks["equity_shock_pct"] = eq_shock_pct
            valuation_method = "Equity Price Shock: V_stressed = V_base * (1 + shock_pct)"

        elif asset.asset_type == "government_bond":
            # Duration-based sensitivity to interest rates only
            dur = asset.duration if asset.duration is not None else 4.0
            dy = rate_shock_bps / 10000.0  # Convert bps to decimal
            delta_p_pct = -dur * dy
            stressed_val = max(0.0, base_val * (1.0 + delta_p_pct))
            applied_shocks["interest_rate_shock_bps"] = rate_shock_bps
            applied_shocks["duration"] = dur
            valuation_method = "Duration Sensitivity: V_stressed = V_base * (1 - Duration * Delta_y_rates)"

        elif asset.asset_type == "corporate_bond":
            # Duration sensitivity to interest rates + credit spread widening
            dur = asset.duration if asset.duration is not None else 4.0
            dy = rate_shock_bps / 10000.0
            ds = spread_shock_bps / 10000.0
            delta_p_pct = -dur * (dy + ds)
            stressed_val = max(0.0, base_val * (1.0 + delta_p_pct))
            applied_shocks["interest_rate_shock_bps"] = rate_shock_bps
            applied_shocks["credit_spread_shock_bps"] = spread_shock_bps
            applied_shocks["duration"] = dur
            valuation_method = "Duration & Spread Sensitivity: V_stressed = V_base * (1 - Duration * (Delta_y_rates + Delta_s_spread))"

        elif asset.asset_type == "loan":
            # Baseline expected loss stressed by multiplier
            pd = asset.baseline_pd if asset.baseline_pd is not None else 0.03
            lgd = asset.lgd if asset.lgd is not None else 0.45
            stress_mult = 1.0 + (loan_loss_shock_pct / 100.0)
            stressed_loss_rate = min(1.0, pd * lgd * stress_mult)
            stressed_val = max(0.0, base_val * (1.0 - stressed_loss_rate))
            applied_shocks["loan_loss_shock_pct"] = loan_loss_shock_pct
            applied_shocks["baseline_pd"] = pd
            applied_shocks["lgd"] = lgd
            applied_shocks["effective_loss_rate"] = round(stressed_loss_rate, 4)
            valuation_method = "Loan Credit Impairment: V_stressed = V_base * (1 - (PD * LGD * (1 + shock_pct)))"

        elif asset.asset_type == "derivative":
            # Simplified DV01 valuation of an Interest Rate Swap (IRS)
            # DV01 (Dollar Value of an 01) is expressed as currency ($) change per 1 basis point (0.0001) shift in benchmark rates.
            # Normalized sensitivity (% per bp) = (dv01_dollar / base_val) * 100
            #
            # Directional P&L Mechanics:
            # - Pay-Fixed Swap (Payer Hedge): Pays fixed rate, receives floating rate.
            #   Rising rates (+Delta_y_rates) increase floating cashflows received -> MtM value increases (gain).
            #   Falling rates (-Delta_y_rates) reduce floating cashflows received -> MtM value decreases (loss).
            # - Receive-Fixed Swap (Receiver): Receives fixed rate, pays floating rate.
            #   Rising rates (+Delta_y_rates) increase floating cashflows paid -> MtM value decreases (loss).
            #   Falling rates (-Delta_y_rates) decrease floating cashflows paid -> MtM value increases (gain).
            direction = (asset.pay_receive or "pay_fixed").lower().strip()
            dur = asset.duration if asset.duration is not None else 4.2
            notional = asset.notional if asset.notional is not None else (base_val * 10.0)

            # Currency DV01 ($/bp): Currency value change per basis point
            if asset.dv01 is not None and asset.dv01 > 0:
                dv01_dollar = float(asset.dv01)
            else:
                dv01_dollar = round(notional * dur * 0.0001, 2)

            # Directional valuation change
            if direction in ("pay_fixed", "payer"):
                delta_v = dv01_dollar * rate_shock_bps
                direction_label = "Pay-Fixed (Payer Macro Hedge)"
            else:
                delta_v = -dv01_dollar * rate_shock_bps
                direction_label = "Receive-Fixed (Receiver)"

            stressed_val = max(0.0, round(base_val + delta_v, 2))
            applied_shocks["interest_rate_shock_bps"] = rate_shock_bps
            applied_shocks["notional"] = notional
            applied_shocks["direction"] = direction_label
            applied_shocks["dv01_dollar"] = dv01_dollar
            applied_shocks["normalized_sensitivity_pct_per_bp"] = (
                round((dv01_dollar / base_val) * 100.0, 4) if base_val > 0 else 0.0
            )
            valuation_method = (
                f"Linear DV01 Swap Valuation ({direction_label}): V_stressed = V_base {'+' if direction in ('pay_fixed', 'payer') else '-'} "
                f"(DV01_$ * Delta_y_rates_bps)"
            )

        else:
            stressed_val = base_val
            valuation_method = "Unstressed / No applicable rule"

        abs_loss = round(base_val - stressed_val, 2)
        pct_loss = round((abs_loss / base_val) * 100.0, 2) if base_val > 0 else 0.0

        return AssetStressResult(
            asset_id=asset.asset_id,
            asset_name=asset.asset_name,
            asset_type=asset.asset_type,
            baseline_value=round(base_val, 2),
            stressed_value=round(stressed_val, 2),
            absolute_loss=round(abs_loss, 2),
            percentage_loss=pct_loss,
            loss_contribution_pct=0.0,  # Computed at portfolio consolidation
            applied_shocks=applied_shocks,
            valuation_method=valuation_method,
        )

    def run_stress_test(
        self,
        portfolio: Portfolio,
        scenario_id: Optional[str] = None,
        custom_parameters: Optional[Dict[str, float]] = None,
        custom_name: str = "Custom Shock Scenario",
        custom_category: str = "Custom",
    ) -> StressTestResult:
        """Execute portfolio-level stress test.

        Args:
            portfolio: Validated Portfolio instance.
            scenario_id: ID of preconfigured scenario from YAML, if using predefined.
            custom_parameters: Dictionary of explicit shock parameters, if running custom.
            custom_name: Optional descriptive name for custom scenario.
            custom_category: Optional category for custom scenario.

        Returns:
            StressTestResult with portfolio and asset-level metrics.
        """
        if scenario_id:
            if scenario_id not in self.scenarios:
                raise ValueError(
                    f"Unknown scenario ID '{scenario_id}'. Available: {list(self.scenarios.keys())}"
                )
            scen_def = self.scenarios[scenario_id]
            name = scen_def.get("name", scenario_id)
            category = scen_def.get("category", "General")
            params = scen_def.get("parameters", {})
        elif custom_parameters is not None:
            name = custom_name
            category = custom_category
            params = custom_parameters
            scenario_id = "custom_scenario"
        else:
            raise ValueError("Must provide either 'scenario_id' or 'custom_parameters'")

        asset_results: List[AssetStressResult] = []
        for asset in portfolio.assets:
            res = self.stress_asset(asset=asset, parameters=params)
            asset_results.append(res)

        base_total = sum(a.baseline_value for a in asset_results)
        stressed_total = sum(a.stressed_value for a in asset_results)
        total_abs_loss = round(base_total - stressed_total, 2)
        total_pct_loss = round((total_abs_loss / base_total) * 100.0, 2) if base_total > 0 else 0.0

        # Calculate asset-level contribution to total portfolio loss
        for a in asset_results:
            if total_abs_loss > 0:
                a.loss_contribution_pct = round((a.absolute_loss / total_abs_loss) * 100.0, 2)
            else:
                a.loss_contribution_pct = 0.0

        return StressTestResult(
            scenario_id=scenario_id,
            scenario_name=name,
            category=category,
            baseline_portfolio_value=round(base_total, 2),
            stressed_portfolio_value=round(stressed_total, 2),
            total_absolute_loss=round(total_abs_loss, 2),
            total_percentage_loss=total_pct_loss,
            asset_results=asset_results,
            scenario_parameters=params,
            disclaimer=DISCLAIMER_TEXT,
        )
