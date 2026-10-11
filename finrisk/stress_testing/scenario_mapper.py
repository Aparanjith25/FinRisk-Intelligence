"""News-to-Scenario Stress Testing Mapping Module.

Provides transparent, explainable decision-support mapping from categorized
financial risk signals to established portfolio stress test scenarios:
- Geopolitical Escalation & Trade Shock (`geopolitical_escalation`)
- Macroeconomic Stagflation Shock (`macroeconomic_stagflation`)
- Systemic Credit Crunch & Liquidity Contagion (`credit_crunch`)
- Mild Cyclical Recession (`mild_recession`)

Features:
- Deterministic rule-based mapping with explainable rationales.
- Strict record deduplication to prevent double-counting.
- Safe isolation of unmapped operational events (M&A, Product Launch, Other).
- User-in-the-loop requirement: produces recommendations for review, never
  silently executes simulated shocks.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Known preconfigured scenario metadata matching configs/stress_scenarios.yaml
SCENARIO_METADATA: Dict[str, Dict[str, str]] = {
    "geopolitical_escalation": {
        "id": "geopolitical_escalation",
        "name": "Geopolitical Escalation & Trade Shock",
        "category": "Geopolitical",
        "description": "Escalation in geopolitical conflicts, aggressive tariffs, trade restrictions, and energy supply disruptions.",
    },
    "macroeconomic_stagflation": {
        "id": "macroeconomic_stagflation",
        "name": "Macroeconomic Stagflation Shock",
        "category": "Macroeconomic",
        "description": "Persistent inflation driving aggressive central bank rate hikes combined with negative GDP contraction.",
    },
    "credit_crunch": {
        "id": "credit_crunch",
        "name": "Systemic Credit Crunch & Liquidity Contagion",
        "category": "Credit Event",
        "description": "Surge in corporate defaults, rating downgrades, covenant breaches, and widening credit spreads.",
    },
    "mild_recession": {
        "id": "mild_recession",
        "name": "Mild Cyclical Recession",
        "category": "Macroeconomic",
        "description": "Moderate slowdown in consumer spending and modest tightening of corporate credit.",
    },
}

MAPPER_DISCLAIMER = (
    "Decision Support Only: News events provide qualitative risk context and do NOT "
    "automatically calibrate quantitative loss percentages or execute portfolio tests. "
    "Portfolio managers must review the recommendation, inspect contributing articles, "
    "and explicitly confirm execution."
)


@dataclass
class ScenarioMatch:
    """Outcome of mapping an individual risk signal to a stress test scenario."""

    record_id: str
    scenario_id: Optional[str]
    scenario_name: str
    category: str
    relevance_score: float  # In [0.0, 1.0]
    rationale: str
    is_mapped: bool
    unmapped_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregatedScenarioRecommendation:
    """Consolidated recommendation for a stress test scenario backed by multiple news signals."""

    scenario_id: str
    scenario_name: str
    category: str
    trigger_count: int
    average_impact: float
    average_sentiment: float
    composite_score: float  # In [0.0, 1.0]
    rationale: str
    contributing_signal_ids: List[str] = field(default_factory=list)
    contributing_signals_summary: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioMappingSummary:
    """Complete summary of news-to-scenario decision support analysis."""

    total_signals_analyzed: int
    unique_signals_analyzed: int
    mapped_signal_count: int
    unmapped_signal_count: int
    unmapped_breakdown: Dict[str, int]
    recommendations: List[AggregatedScenarioRecommendation]
    top_recommendation: Optional[AggregatedScenarioRecommendation]
    disclaimer: str = MAPPER_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["recommendations"] = [r.to_dict() for r in self.recommendations]
        res["top_recommendation"] = self.top_recommendation.to_dict() if self.top_recommendation else None
        return res


def map_signal_to_scenario(signal: Dict[str, Any]) -> ScenarioMatch:
    """Map a single financial risk signal into an applicable stress test scenario.

    Args:
        signal: Dictionary representing a structured RiskSignal.

    Returns:
        ScenarioMatch object with transparent rationale and relevance score.
    """
    if not isinstance(signal, dict):
        return ScenarioMatch(
            record_id="invalid",
            scenario_id=None,
            scenario_name="Unmapped",
            category="Invalid",
            relevance_score=0.0,
            rationale="Malformed signal input (non-dict).",
            is_mapped=False,
            unmapped_reason="Malformed Non-Dict Input",
        )

    record_id = str(signal.get("record_id") or "unspecified").strip()
    event_type = str(signal.get("event_type") or "Other").strip()
    try:
        raw_sentiment = float(signal.get("sentiment_score", 0.0))
        sentiment_score = max(-1.0, min(1.0, raw_sentiment))
    except (ValueError, TypeError):
        sentiment_score = 0.0

    try:
        raw_impact = int(signal.get("impact_score", 0))
        impact_score = max(0, min(10, raw_impact))
    except (ValueError, TypeError):
        impact_score = 0

    text = str(signal.get("text", "")).lower()

    # Rule 1: Geopolitical events
    if event_type == "Geopolitical":
        if sentiment_score < 0.15 and impact_score >= 3:
            scen_meta = SCENARIO_METADATA["geopolitical_escalation"]
            relevance = min(1.0, round(0.50 + 0.04 * impact_score + 0.20 * abs(min(0.0, sentiment_score)), 2))
            rationale = (
                f"Geopolitical risk event with adverse sentiment ({sentiment_score:+.2f}) "
                f"and elevated impact score ({impact_score}/10) triggers '{scen_meta['name']}'."
            )
            return ScenarioMatch(
                record_id=record_id,
                scenario_id="geopolitical_escalation",
                scenario_name=scen_meta["name"],
                category=scen_meta["category"],
                relevance_score=relevance,
                rationale=rationale,
                is_mapped=True,
            )
        else:
            return ScenarioMatch(
                record_id=record_id,
                scenario_id=None,
                scenario_name="Unmapped",
                category="Geopolitical",
                relevance_score=0.0,
                rationale="Geopolitical event has low impact (< 3) or positive tone; no systemic escalation indicated.",
                is_mapped=False,
                unmapped_reason="Low Severity / Benign Tone",
            )

    # Rule 2: Macroeconomic events
    elif event_type == "Macroeconomic":
        # Check for inflation, rate hikes, stagflation, or high impact
        stagflation_indicators = (
            "stagflation", "inflation", "cpi", "rate hike", "hikes", "tightening",
            "borrowing costs", "yield curve", "treasury yields"
        )
        is_stagflation = any(k in text for k in stagflation_indicators)

        slowdown_indicators = (
            "recession", "gdp contraction", "slowdown", "cooling", "rate cut", "cuts",
            "easing", "unemployment"
        )
        is_slowdown = any(k in text for k in slowdown_indicators)

        if (is_stagflation or impact_score >= 6) and sentiment_score <= 0.10:
            scen_meta = SCENARIO_METADATA["macroeconomic_stagflation"]
            relevance = min(1.0, round(0.55 + 0.04 * impact_score + 0.15 * abs(min(0.0, sentiment_score)), 2))
            rationale = (
                f"Macroeconomic monetary tightening / inflation pressure with impact score "
                f"{impact_score}/10 maps to '{scen_meta['name']}'."
            )
            return ScenarioMatch(
                record_id=record_id,
                scenario_id="macroeconomic_stagflation",
                scenario_name=scen_meta["name"],
                category=scen_meta["category"],
                relevance_score=relevance,
                rationale=rationale,
                is_mapped=True,
            )
        elif is_slowdown or (impact_score < 6 and sentiment_score < 0.10):
            scen_meta = SCENARIO_METADATA["mild_recession"]
            relevance = min(1.0, round(0.45 + 0.05 * impact_score, 2))
            rationale = (
                f"Macroeconomic cyclical slowdown / moderation indicators with impact score "
                f"{impact_score}/10 suggest '{scen_meta['name']}'."
            )
            return ScenarioMatch(
                record_id=record_id,
                scenario_id="mild_recession",
                scenario_name=scen_meta["name"],
                category=scen_meta["category"],
                relevance_score=relevance,
                rationale=rationale,
                is_mapped=True,
            )
        else:
            return ScenarioMatch(
                record_id=record_id,
                scenario_id=None,
                scenario_name="Unmapped",
                category="Macroeconomic",
                relevance_score=0.0,
                rationale="Macroeconomic event reflects benign or positive growth; no downside shock triggered.",
                is_mapped=False,
                unmapped_reason="Benign Macro Expansion",
            )

    # Rule 3: Credit Events
    elif event_type == "Credit Event":
        if sentiment_score < 0.15 and impact_score >= 3:
            scen_meta = SCENARIO_METADATA["credit_crunch"]
            relevance = min(1.0, round(0.60 + 0.04 * impact_score + 0.10 * abs(min(0.0, sentiment_score)), 2))
            rationale = (
                f"Credit distress / default / downgrade signal with impact score "
                f"{impact_score}/10 maps to '{scen_meta['name']}'."
            )
            return ScenarioMatch(
                record_id=record_id,
                scenario_id="credit_crunch",
                scenario_name=scen_meta["name"],
                category=scen_meta["category"],
                relevance_score=relevance,
                rationale=rationale,
                is_mapped=True,
            )
        else:
            return ScenarioMatch(
                record_id=record_id,
                scenario_id=None,
                scenario_name="Unmapped",
                category="Credit Event",
                relevance_score=0.0,
                rationale="Credit event exhibits minor severity (< 3) or favorable rating outlook.",
                is_mapped=False,
                unmapped_reason="Low Severity Credit Item",
            )

    # Rule 4: Microeconomic and Operational Events (M&A, Product Launch, Other)
    else:
        unmapped_category_names = {
            "Merger/Acquisition": "Corporate Transaction (M&A)",
            "Product Launch": "Commercial / Product Launch",
            "Other": "Routine Operational Disclosure",
        }
        cat_desc = unmapped_category_names.get(event_type, event_type)
        return ScenarioMatch(
            record_id=record_id,
            scenario_id=None,
            scenario_name="Unmapped",
            category=event_type,
            relevance_score=0.0,
            rationale=(
                f"Microeconomic event ('{cat_desc}') is intentionally excluded from "
                f"macro portfolio stress scenarios to prevent false-positive shock contagion."
            ),
            is_mapped=False,
            unmapped_reason=f"{cat_desc} (Microeconomic/Operational)",
        )


def recommend_scenarios_from_signals(signals: List[Dict[str, Any]]) -> ScenarioMappingSummary:
    """Analyze a batch of risk signals, deduplicate them, and produce ranked scenario recommendations.

    Args:
        signals: List of dictionary records representing RiskSignal objects.

    Returns:
        ScenarioMappingSummary with ranked recommendations and explainable rationales.
    """
    if not signals:
        return ScenarioMappingSummary(
            total_signals_analyzed=0,
            unique_signals_analyzed=0,
            mapped_signal_count=0,
            unmapped_signal_count=0,
            unmapped_breakdown={},
            recommendations=[],
            top_recommendation=None,
            disclaimer=MAPPER_DISCLAIMER,
        )

    # 1. Deduplicate by record_id (or content hash) to prevent double-counting
    import hashlib

    seen_ids: set[str] = set()
    unique_signals: List[Dict[str, Any]] = []
    for s in signals:
        if not isinstance(s, dict):
            continue
        rid = str(s.get("record_id") or "").strip()
        effective_id = rid or hashlib.sha256(str(s.get("text", "")).encode("utf-8")).hexdigest()[:16]
        if effective_id in seen_ids:
            continue
        seen_ids.add(effective_id)
        unique_signals.append(s)

    # 2. Map each unique signal
    matches: List[ScenarioMatch] = []
    unmapped_breakdown: Dict[str, int] = {}
    grouped_matches: Dict[str, List[tuple[ScenarioMatch, Dict[str, Any]]]] = {
        k: [] for k in SCENARIO_METADATA.keys()
    }

    for sig in unique_signals:
        match = map_signal_to_scenario(sig)
        matches.append(match)
        if match.is_mapped and match.scenario_id in grouped_matches:
            grouped_matches[match.scenario_id].append((match, sig))
        else:
            reason = match.unmapped_reason or "Other Unmapped"
            unmapped_breakdown[reason] = unmapped_breakdown.get(reason, 0) + 1

    mapped_count = sum(1 for m in matches if m.is_mapped)
    unmapped_count = len(matches) - mapped_count

    # 3. Construct aggregated recommendations for scenarios with >= 1 trigger
    recommendations: List[AggregatedScenarioRecommendation] = []
    for scen_id, group in grouped_matches.items():
        if not group:
            continue
        scen_meta = SCENARIO_METADATA[scen_id]
        trigger_count = len(group)
        impacts = [float(sig.get("impact_score", 0)) for _, sig in group]
        sentiments = [float(sig.get("sentiment_score", 0.0)) for _, sig in group]

        avg_impact = round(sum(impacts) / trigger_count, 2)
        avg_sentiment = round(sum(sentiments) / trigger_count, 4)

        # Composite score balancing frequency, severity (impact), and negative sentiment
        freq_component = min(1.0, trigger_count / max(1, mapped_count))
        impact_component = min(1.0, avg_impact / 10.0)
        sentiment_component = min(1.0, abs(min(0.0, avg_sentiment)))

        composite_score = round(
            min(1.0, 0.40 * freq_component + 0.40 * impact_component + 0.20 * sentiment_component),
            4,
        )

        contributing_signal_ids = [m.record_id for m, _ in group]
        contributing_summary = [
            {
                "record_id": m.record_id,
                "source": sig.get("source", "Unknown"),
                "company": sig.get("company"),
                "event_type": sig.get("event_type"),
                "impact_score": sig.get("impact_score"),
                "sentiment_score": sig.get("sentiment_score"),
                "headline_excerpt": (sig.get("text") or "")[:90] + ("..." if len(sig.get("text") or "") > 90 else ""),
            }
            for m, sig in group
        ]

        rationale = (
            f"Supported by {trigger_count} independent risk signal(s) with an average impact of "
            f"{avg_impact}/10 and average sentiment score of {avg_sentiment:+.2f}. "
            f"Qualitative narrative indicates {scen_meta['name'].lower()} dynamics."
        )

        rec = AggregatedScenarioRecommendation(
            scenario_id=scen_id,
            scenario_name=scen_meta["name"],
            category=scen_meta["category"],
            trigger_count=trigger_count,
            average_impact=avg_impact,
            average_sentiment=avg_sentiment,
            composite_score=composite_score,
            rationale=rationale,
            contributing_signal_ids=contributing_signal_ids,
            contributing_signals_summary=contributing_summary,
        )
        recommendations.append(rec)

    # Rank recommendations by composite score descending, then trigger count descending
    recommendations.sort(key=lambda r: (r.composite_score, r.trigger_count), reverse=True)
    top_rec = recommendations[0] if recommendations else None

    return ScenarioMappingSummary(
        total_signals_analyzed=len(signals),
        unique_signals_analyzed=len(unique_signals),
        mapped_signal_count=mapped_count,
        unmapped_signal_count=unmapped_count,
        unmapped_breakdown=unmapped_breakdown,
        recommendations=recommendations,
        top_recommendation=top_rec,
        disclaimer=MAPPER_DISCLAIMER,
    )
