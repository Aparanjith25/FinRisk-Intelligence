"""Financial event classification module.

Provides a modular classifier interface with a transparent, deterministic baseline
heuristic for offline testing, while preserving an extensible architecture for future
trained classification models.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Canonical event categories
EVENT_CATEGORIES = (
    "Geopolitical",
    "Macroeconomic",
    "Credit Event",
    "Merger/Acquisition",
    "Product Launch",
    "Other",
)

# Comprehensive baseline keyword and phrase regex patterns
BASELINE_EVENT_REGEXES: Dict[str, List[str]] = {
    "Geopolitical": [
        r"sanctions?",
        r"embargo(?:es)?",
        r"tariffs?",
        r"trade\s+wars?",
        r"military(?:\s+action|\s+conflict)?",
        r"war(?:fare)?",
        r"geopolitical(?:\s+tensions?|\s+risks?)?",
        r"opec(?:\+)?",
        r"export\s+(?:controls?|bans?|restrictions?|quotas?)",
        r"border\s+(?:conflicts?|disputes?)",
        r"treat(?:y|ies)",
        r"foreign\s+polic(?:y|ies)",
        r"sovereign\s+risks?",
        r"national\s+securit(?:y|ies)",
        r"regulator(?:y|ies)?\s+(?:inquir(?:y|ies)|reviews?|actions?|penalt(?:y|ies)|fines?|scrutiny)",
        r"environmental\s+fines?",
        r"antitrust(?:\s+inquir(?:y|ies)|\s+regulat(?:ory|ion))?",
    ],
    "Macroeconomic": [
        r"inflation(?:ary)?",
        r"cpi",
        r"interest\s+rates?",
        r"central\s+banks?",
        r"federal\s+reserve",
        r"the\s+fed\b",
        r"gdp(?:\s+growth)?",
        r"unemployment(?:\s+rate)?",
        r"monetary\s+polic(?:y|ies)",
        r"recession(?:ary)?",
        r"yield\s+curves?",
        r"treasury\s+yields?",
        r"stagflation",
        r"rate\s+(?:hikes?|cuts?|decisions?)",
        r"credit\s+cycles?",
        r"borrowing\s+costs?",
        r"freight\s+rates?",
        r"quantitative\s+(?:easing|tightening)",
        r"labor\s+markets?",
    ],
    "Credit Event": [
        r"defaults?",
        r"defaulted",
        r"defaulting",
        r"bankruptc(?:y|ies)",
        r"debt\s+restructur(?:ing|ed|e)",
        r"(?:credit\s+|rating\s+)?downgrades?",
        r"downgraded",
        r"(?:covenant\s+)?breach(?:es)?",
        r"insolven(?:cy|t)",
        r"liquidations?",
        r"credit\s+default\s+swaps?",
        r"\bcds\b",
        r"distressed\s+debts?",
        r"chapter\s+11",
        r"loan\s+loss\s+provisions?",
        r"reserve\s+provisions?",
        r"credit\s+spreads?(?:\s+widening|\s+blowouts?)?",
        r"non-performing\s+loans?",
        r"credit\s+ratings?",
        r"coupon\s+defaults?",
        r"loan\s+defaults?",
    ],
    "Merger/Acquisition": [
        r"acquisitions?",
        r"acquires?",
        r"acquired",
        r"acquiring",
        r"buyouts?",
        r"takeovers?",
        r"takeover\s+agreements?",
        r"mergers?",
        r"merging",
        r"tender\s+offers?",
        r"divestitures?",
        r"stake\s+purchases?",
        r"merger\s+agreements?",
        r"hostile\s+bids?",
        r"private\s+equity\s+(?:buyouts?|deals?)",
        r"spinoffs?",
        r"spun\s+off",
    ],
    "Product Launch": [
        r"products?\s+launch(?:es|ed|ing)?",
        r"unveils?",
        r"unveiled",
        r"unveiling",
        r"new\s+platforms?",
        r"commercial\s+availabilit(?:y|ies)",
        r"clinical\s+trials?",
        r"fda\s+approvals?",
        r"patents?(?:\s+granted)?",
        r"granted\s+(?:key\s+)?patents?",
        r"new\s+models?",
        r"rolls?\s+out",
        r"rolled\s+out",
        r"rolling\s+out",
        r"introduces?\s+new",
    ],
}


@dataclass
class EventClassificationResult:
    """Output of financial event classification."""

    category: str  # One of EVENT_CATEGORIES
    confidence: float  # In [0.0, 1.0]
    matched_keywords: List[str] = field(default_factory=list)
    competing_categories: Dict[str, List[str]] = field(default_factory=dict)
    classifier_type: str = "RuleBasedEventClassifier (Provisional Baseline)"
    rationale: str = ""
    is_provisional: bool = True  # Explicit flag indicating baseline heuristic status


class BaseEventClassifier(ABC):
    """Abstract base class for financial event classification."""

    @abstractmethod
    def classify(self, text: str) -> EventClassificationResult:
        """Classify a single financial text into an event category."""
        pass

    def classify_batch(self, texts: List[str]) -> List[EventClassificationResult]:
        """Classify a batch of texts."""
        return [self.classify(t) for t in texts]


class RuleBasedEventClassifier(BaseEventClassifier):
    """Transparent, deterministic rule-based event classifier.

    NOTE: This is a provisional baseline designed for offline pipeline operation
    and deterministic testing. It relies on curated financial vocabulary patterns
    and is NOT equivalent to a trained, calibrated event classifier.
    """

    def __init__(self, patterns: Optional[Dict[str, List[str]]] = None) -> None:
        """Initialize with custom regex patterns or defaults."""
        raw_patterns = patterns or BASELINE_EVENT_REGEXES
        self._compiled: Dict[str, List[re.Pattern]] = {}
        for cat, regex_list in raw_patterns.items():
            compiled_list = []
            for pattern_str in regex_list:
                compiled_list.append(re.compile(rf"\b{pattern_str}\b", re.IGNORECASE))
            self._compiled[cat] = compiled_list

    def classify(self, text: str) -> EventClassificationResult:
        """Classify text using deterministic regular expression pattern matching.

        Args:
            text: Financial text string.

        Returns:
            EventClassificationResult with matched category, matched keywords,
            confidence, competing candidate categories, and explainable rationale.
        """
        if not text or not str(text).strip():
            return EventClassificationResult(
                category="Other",
                confidence=0.0,
                matched_keywords=[],
                competing_categories={},
                classifier_type="RuleBasedEventClassifier (Provisional Baseline)",
                rationale="Empty text input provided; defaulted to 'Other'.",
                is_provisional=True,
            )

        clean_text = str(text).strip()
        category_matches: Dict[str, List[str]] = {}

        for category, regex_list in self._compiled.items():
            matched_for_cat: Set[str] = set()
            for rx in regex_list:
                found = rx.findall(clean_text)
                if found:
                    for item in found:
                        matched_for_cat.add(item.lower() if isinstance(item, str) else str(item).lower())
            if matched_for_cat:
                category_matches[category] = sorted(matched_for_cat)

        if not category_matches:
            return EventClassificationResult(
                category="Other",
                confidence=0.35,
                matched_keywords=[],
                competing_categories={},
                classifier_type="RuleBasedEventClassifier (Provisional Baseline)",
                rationale="No category keywords matched; classified as 'Other'. (Provisional baseline).",
                is_provisional=True,
            )

        # Sort categories by number of matched distinct terms descending
        sorted_categories = sorted(
            category_matches.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )

        best_category, best_matches = sorted_categories[0]
        match_count = len(best_matches)

        # Competing categories (ambiguity detection)
        competing: Dict[str, List[str]] = {
            cat: kw for cat, kw in sorted_categories[1:]
        }

        # Confidence heuristic:
        # If there are competing categories with equal matches, penalize confidence for ambiguity
        if competing and len(list(competing.values())[0]) == match_count:
            # Ambiguous tie: first candidate chosen deterministically, lower confidence
            confidence = round(min(0.70, 0.45 + 0.10 * match_count), 2)
            ambiguity_note = f" Ambiguous: competing matches found for {', '.join(competing.keys())}."
        else:
            confidence = round(min(0.90, 0.50 + 0.15 * match_count), 2)
            ambiguity_note = ""

        rationale = (
            f"Matched {match_count} keyword(s) for '{best_category}': {', '.join(best_matches)}.{ambiguity_note} "
            f"(Provisional baseline heuristic; not a trained model)."
        )

        return EventClassificationResult(
            category=best_category,
            confidence=confidence,
            matched_keywords=best_matches,
            competing_categories=competing,
            classifier_type="RuleBasedEventClassifier (Provisional Baseline)",
            rationale=rationale,
            is_provisional=True,
        )
