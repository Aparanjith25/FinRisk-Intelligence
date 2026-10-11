"""Quantitative risk engine for metric computation and risk scoring."""

from finrisk.risk_engine.engine import (
    RISK_SIGNAL_FIELDS,
    RiskEngine,
    RiskSignal,
)
from finrisk.risk_engine.impact_scorer import (
    EVENT_BASE_IMPACT,
    ImpactScoreResult,
    compute_impact_score,
)

__all__ = [
    "EVENT_BASE_IMPACT",
    "ImpactScoreResult",
    "compute_impact_score",
    "RISK_SIGNAL_FIELDS",
    "RiskEngine",
    "RiskSignal",
]
