"""Financial impact scoring module.

Computes an explainable integer impact score from 1 to 10 based on event category,
sentiment magnitude, identified entities, and quantitative text evidence.
NOTE: This is a transparent rule-based heuristic, not a calibrated econometric prediction.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Base impact weights by event category (1 to 10 scale baseline)
EVENT_BASE_IMPACT: Dict[str, int] = {
    "Credit Event": 5,        # Solvency, default, and covenant distress have high direct risk
    "Geopolitical": 4,        # Macro/supply shocks carry systemic uncertainty
    "Macroeconomic": 4,       # Rate, inflation, and monetary shifts affect portfolio-wide discount rates
    "Merger/Acquisition": 3,  # Significant balance-sheet and execution adjustments
    "Product Launch": 2,      # Localized microeconomic commercial impact
    "Other": 1,               # Unclassified or low-specificity events
}

# Regex for detecting quantitative financial figures in text
QUANTITATIVE_EVIDENCE_PATTERN = re.compile(
    r"(\$\d+|\b\d+\s*percent\b|\b\d+(\.\d+)?%\b|\b\d+\s*bps\b|\b\d+\s*million\b|\b\d+\s*billion\b)",
    re.IGNORECASE,
)


@dataclass
class ImpactScoreResult:
    """Detailed output of impact score calculation."""

    score: int  # Integer in [1, 10]
    breakdown: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    limitations: str = (
        "Provisional heuristic impact score based on rule-based weights. "
        "This is not a calibrated market-impact prediction or econometric forecast."
    )


def compute_impact_score(
    event_category: Optional[str],
    sentiment_score: Optional[float] = None,
    sentiment_confidence: Optional[float] = None,
    company: Optional[str] = None,
    text: Optional[str] = None,
) -> ImpactScoreResult:
    """Calculate an explainable impact score between 1 and 10.

    Args:
        event_category: Event category string (e.g. 'Credit Event', 'Macroeconomic').
        sentiment_score: Continuous sentiment score in [-1.0, 1.0], or None if unavailable.
        sentiment_confidence: Sentiment model confidence in [0.0, 1.0], or None.
        company: Named company or entity, or None.
        text: Underlying text snippet for quantitative evidence detection, or None.

    Returns:
        ImpactScoreResult with integer score in [1, 10], component breakdown, and rationale.
    """
    rationale_parts = []
    breakdown: Dict[str, Any] = {}

    # 1. Base Impact from Event Category
    cat_clean = str(event_category).strip() if event_category else "Other"
    base_score = EVENT_BASE_IMPACT.get(cat_clean, 1)
    breakdown["event_category"] = cat_clean
    breakdown["base_score"] = base_score
    rationale_parts.append(f"Base score {base_score} for category '{cat_clean}'")

    running_score = float(base_score)

    # 2. Sentiment Magnitude Adjustment
    sentiment_adj = 0
    if sentiment_score is not None:
        try:
            s_val = float(sentiment_score)
            if s_val <= -0.6:
                sentiment_adj = 3
                rationale_parts.append(f"+3 for severe negative sentiment ({s_val:.2f})")
            elif s_val <= -0.2:
                sentiment_adj = 2
                rationale_parts.append(f"+2 for moderate negative sentiment ({s_val:.2f})")
            elif s_val < 0.0:
                sentiment_adj = 1
                rationale_parts.append(f"+1 for mild negative sentiment ({s_val:.2f})")
            elif s_val >= 0.6:
                sentiment_adj = 2
                rationale_parts.append(f"+2 for high positive catalyst sentiment ({s_val:.2f})")
            elif s_val >= 0.2:
                sentiment_adj = 1
                rationale_parts.append(f"+1 for moderate positive sentiment ({s_val:.2f})")
            else:
                sentiment_adj = 0
                rationale_parts.append(f"+0 for neutral sentiment ({s_val:.2f})")
        except (ValueError, TypeError):
            sentiment_adj = 0
            rationale_parts.append("+0 (invalid sentiment value provided)")
    else:
        rationale_parts.append("+0 (sentiment score missing/unavailable)")

    breakdown["sentiment_adjustment"] = sentiment_adj
    running_score += sentiment_adj

    # 3. Entity Specificity Adjustment
    entity_adj = 0
    if company and str(company).strip():
        entity_adj = 1
        rationale_parts.append(f"+1 for identified company exposure ('{str(company).strip()}')")
    else:
        rationale_parts.append("+0 (no specific company identified)")

    breakdown["entity_adjustment"] = entity_adj
    running_score += entity_adj

    # 4. Quantitative Evidence Adjustment
    evidence_adj = 0
    if text and QUANTITATIVE_EVIDENCE_PATTERN.search(str(text)):
        evidence_adj = 1
        rationale_parts.append("+1 for quantitative figures/metrics found in text")
    else:
        rationale_parts.append("+0 (no quantitative figures detected)")

    breakdown["evidence_adjustment"] = evidence_adj
    running_score += evidence_adj

    # 5. Low Confidence Penalty
    confidence_adj = 0
    if sentiment_confidence is not None:
        try:
            conf_val = float(sentiment_confidence)
            if conf_val < 0.50:
                confidence_adj = -1
                rationale_parts.append(f"-1 penalty for low model confidence ({conf_val:.2f})")
        except (ValueError, TypeError):
            pass

    breakdown["confidence_adjustment"] = confidence_adj
    running_score += confidence_adj

    # 6. Clamp strictly between 1 and 10
    final_score = int(max(1, min(10, round(running_score))))
    breakdown["raw_calculated_score"] = round(running_score, 2)
    breakdown["final_score"] = final_score

    rationale = "; ".join(rationale_parts) + f" -> Final impact: {final_score}/10"

    return ImpactScoreResult(
        score=final_score,
        breakdown=breakdown,
        rationale=rationale,
    )
