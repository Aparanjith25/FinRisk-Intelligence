"""High-impact event alert detection and recommendation module.

Monitors ingested risk signals, identifies high-impact events exceeding a configurable
threshold (default: impact > 7), correlates them with appropriate stress scenarios via
the ScenarioMapper, and structures human-in-the-loop alerts for portfolio risk managers.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set

from finrisk.stress_testing.scenario_mapper import map_signal_to_scenario

logger = logging.getLogger(__name__)

DEFAULT_HIGH_IMPACT_THRESHOLD = 7


@dataclass
class HighImpactAlert:
    """Represents an actionable alert generated from a high-impact financial event."""

    record_id: str
    event_category: str
    impact_score: int
    sentiment_score: float
    sentiment_label: str
    source: str
    published_at: str
    summary_text: str
    recommended_scenario_id: Optional[str]
    recommended_scenario_name: Optional[str]
    mapping_rationale: str
    is_stress_test_applicable: bool
    requires_human_confirmation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to serializable dictionary."""
        return asdict(self)


def detect_high_impact_alerts(
    signals: List[Dict[str, Any]],
    threshold: int = DEFAULT_HIGH_IMPACT_THRESHOLD,
    deduplicate: bool = True,
) -> List[HighImpactAlert]:
    """Filter and construct high-impact alerts from a list of risk signals.

    Args:
        signals: List of risk signal dictionaries.
        threshold: Minimum impact threshold (alerts trigger strictly for impact > threshold).
        deduplicate: Whether to suppress duplicate signals by record_id and text signature.

    Returns:
        List of HighImpactAlert instances sorted by impact_score (descending).
    """
    alerts: List[HighImpactAlert] = []
    seen_keys: Set[str] = set()

    for s in signals:
        if not isinstance(s, dict):
            continue

        # Safe impact score extraction
        raw_impact = s.get("impact_score")
        if raw_impact is None or raw_impact == "":
            continue
        try:
            impact_score = int(float(raw_impact))
        except (ValueError, TypeError):
            continue

        # Strict threshold condition: impact > threshold
        if impact_score <= threshold:
            continue

        record_id = str(s.get("record_id") or "unspecified").strip()
        text = str(s.get("text") or "").strip()

        # Deduplication
        dedup_key = record_id if record_id != "unspecified" else text[:80].lower()
        if deduplicate and dedup_key:
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

        # Extract other fields safely
        event_category = str(s.get("event_type") or "Other").strip()
        source = str(s.get("source") or "Unknown").strip()
        published_at = str(s.get("published_at") or "Unknown").strip()

        try:
            sentiment_score = float(s.get("sentiment_score", 0.0))
        except (ValueError, TypeError):
            sentiment_score = 0.0

        sentiment_label = str(s.get("sentiment_label") or "neutral").strip()

        # Map to scenario using existing ScenarioMapper logic
        scenario_match = map_signal_to_scenario(s)

        if scenario_match.is_mapped and scenario_match.scenario_id:
            scenario_id = scenario_match.scenario_id
            scenario_name = scenario_match.scenario_name
            rationale = scenario_match.rationale
            is_applicable = True
        else:
            scenario_id = None
            scenario_name = "No Preconfigured Stress Scenario (Operational / Non-Systemic)"
            rationale = (
                scenario_match.unmapped_reason
                or scenario_match.rationale
                or "Event does not map to macro stress testing."
            )
            is_applicable = False

        alert = HighImpactAlert(
            record_id=record_id,
            event_category=event_category,
            impact_score=impact_score,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            source=source,
            published_at=published_at,
            summary_text=text,
            recommended_scenario_id=scenario_id,
            recommended_scenario_name=scenario_name,
            mapping_rationale=rationale,
            is_stress_test_applicable=is_applicable,
            requires_human_confirmation=True,
        )
        alerts.append(alert)

    # Sort alerts by impact_score desc, then sentiment_score asc (most severe first)
    alerts.sort(key=lambda a: (-a.impact_score, a.sentiment_score))
    return alerts
