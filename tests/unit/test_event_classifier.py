"""Unit tests for financial event classification.

Includes labeled synthetic test examples for all six canonical categories
and ambiguous multi-category financial scenarios.
NOTE: These synthetic tests verify deterministic reachability and rule behavior;
they do not represent a statistical benchmark on real-world market distributions.
"""

from __future__ import annotations

import pytest

from finrisk.nlp.event_classifier import (
    EVENT_CATEGORIES,
    BaseEventClassifier,
    EventClassificationResult,
    RuleBasedEventClassifier,
)


def test_event_categories_completeness():
    expected = {"Geopolitical", "Macroeconomic", "Credit Event", "Merger/Acquisition", "Product Launch", "Other"}
    assert set(EVENT_CATEGORIES) == expected


# Synthetic test cases for all six canonical categories
SYNTHETIC_CATEGORY_CASES = [
    (
        "Geopolitical",
        "[SYNTHETIC] Department of Commerce imposes strict export bans and new tariffs on advanced semiconductors.",
        "Geopolitical",
    ),
    (
        "Geopolitical (Plural & Phrase)",
        "[SYNTHETIC] European regulators launched regulatory inquiry into foreign sovereign energy deals.",
        "Geopolitical",
    ),
    (
        "Macroeconomic",
        "[SYNTHETIC] Federal Reserve officials signaled further interest rate cuts as inflation moderates.",
        "Macroeconomic",
    ),
    (
        "Macroeconomic (Labor & GDP)",
        "[SYNTHETIC] National unemployment rate ticked down alongside resilient annualized GDP growth figures.",
        "Macroeconomic",
    ),
    (
        "Credit Event (Plural Provisions)",
        "[SYNTHETIC] Regional lender raised reserve provisions to $80M in anticipation of commercial real estate defaults.",
        "Credit Event",
    ),
    (
        "Credit Event (Rating Downgrade)",
        "[SYNTHETIC] Rating agencies downgraded holding company bonds to junk status following covenant breach.",
        "Credit Event",
    ),
    (
        "Merger/Acquisition",
        "[SYNTHETIC] Industrial conglomerate entered definitive merger agreement to acquire competitor for $4.2B.",
        "Merger/Acquisition",
    ),
    (
        "Merger/Acquisition (Spinoff & Buyout)",
        "[SYNTHETIC] Private equity consortium announced leveraged buyout after planned corporate spinoff was abandoned.",
        "Merger/Acquisition",
    ),
    (
        "Product Launch",
        "[SYNTHETIC] Therapeutics startup unveiled promising Phase 3 clinical trials and filed for FDA approvals.",
        "Product Launch",
    ),
    (
        "Product Launch (Rollout)",
        "[SYNTHETIC] Automaker rolls out next-generation commercial platform for electric delivery fleets.",
        "Product Launch",
    ),
    (
        "Other (Internal Operations)",
        "[SYNTHETIC] Software firm appointed a new Chief People Officer to lead international talent initiatives.",
        "Other",
    ),
    (
        "Other (Routine Facilities)",
        "[SYNTHETIC] Facility management completed scheduled annual server rack cooling maintenance.",
        "Other",
    ),
]


@pytest.mark.parametrize("case_name, text, expected_category", SYNTHETIC_CATEGORY_CASES)
def test_all_categories_reachable(case_name: str, text: str, expected_category: str):
    classifier = RuleBasedEventClassifier()
    result = classifier.classify(text)
    assert result.category == expected_category, (
        f"Failed for case '{case_name}': expected {expected_category}, got {result.category}. "
        f"Rationale: {result.rationale}"
    )
    assert result.is_provisional is True
    assert 0.0 <= result.confidence <= 1.0


def test_ambiguous_scenarios_handling():
    classifier = RuleBasedEventClassifier()

    # Scenario 1: Geopolitical vs Macroeconomic
    # "tariffs" (Geopolitical) vs "inflation" (Macroeconomic)
    text1 = "[SYNTHETIC] Heavy import tariffs triggered consumer price inflation across retail goods."
    res1 = classifier.classify(text1)
    # Both categories matched; competing categories must record the alternative
    assert res1.category in ("Geopolitical", "Macroeconomic")
    assert len(res1.competing_categories) > 0 or len(res1.matched_keywords) > 0
    assert "Ambiguous" in res1.rationale or "Matched" in res1.rationale

    # Scenario 2: Macroeconomic vs Credit Event
    # "interest rates" (Macroeconomic) vs "defaults" & "bankruptcies" (Credit Event)
    text2 = "[SYNTHETIC] Escalating interest rates forced commercial defaults and distressed corporate bankruptcies."
    res2 = classifier.classify(text2)
    # Two credit terms ("defaults", "bankruptcies") vs one macro term ("interest rates") -> Credit Event dominates
    assert res2.category == "Credit Event"
    assert "Macroeconomic" in res2.competing_categories

    # Scenario 3: Merger/Acquisition vs Credit Event
    # "takeover" (M&A) vs "insolvent" (Credit Event)
    text3 = "[SYNTHETIC] Insolvent freight operator saved by unexpected distressed takeover agreement."
    res3 = classifier.classify(text3)
    assert res3.category in ("Merger/Acquisition", "Credit Event")
    assert res3.confidence < 0.90  # Ambiguity dampens confidence


def test_empty_and_whitespace_input():
    classifier = RuleBasedEventClassifier()
    res_empty = classifier.classify("")
    assert res_empty.category == "Other"
    assert res_empty.confidence == 0.0

    res_spaces = classifier.classify("   \t\n  ")
    assert res_spaces.category == "Other"
    assert res_spaces.confidence == 0.0


def test_batch_classification():
    classifier = RuleBasedEventClassifier()
    batch = [
        "Central bank cuts interest rates as inflation cools.",
        "Company filed for Chapter 11 bankruptcy and debt restructuring.",
    ]
    results = classifier.classify_batch(batch)
    assert len(results) == 2
    assert results[0].category == "Macroeconomic"
    assert results[1].category == "Credit Event"


def test_custom_extensibility():
    class CustomClassifier(BaseEventClassifier):
        def classify(self, text: str) -> EventClassificationResult:
            return EventClassificationResult(
                category="Geopolitical",
                confidence=0.99,
                matched_keywords=["custom"],
                classifier_type="CustomModel",
                is_provisional=False,
            )

    custom = CustomClassifier()
    res = custom.classify("Any text string")
    assert res.category == "Geopolitical"
    assert res.classifier_type == "CustomModel"
    assert res.is_provisional is False


def test_event_classifier_refined_patterns_regression():
    classifier = RuleBasedEventClassifier()

    # OPEC oil export quotas
    res_geo = classifier.classify("OPEC export quotas were renewed following ministerial summit.")
    assert res_geo.category == "Geopolitical"

    # Antitrust regulatory inquiries
    res_reg = classifier.classify("Justice Department launched antitrust inquiry into digital ad market.")
    assert res_reg.category == "Geopolitical"

    # Loan defaults and coupon defaults
    res_crd = classifier.classify("Syndicated loan defaults rose alongside missed coupon defaults on bonds.")
    assert res_crd.category == "Credit Event"

    # Takeover agreement
    res_ma = classifier.classify("Board approved takeover agreements after extensive diligence.")
    assert res_ma.category == "Merger/Acquisition"

    # Granted key patents
    res_prod = classifier.classify("Biopharma company was granted key patents for antibody therapies.")
    assert res_prod.category == "Product Launch"


def test_event_classifier_fomc_and_credit_distress_patterns():
    """Verify FOMC, banking supervision, credit distress, and proper retention of Other."""
    classifier = RuleBasedEventClassifier()

    # FOMC / Federal Open Market Committee minutes
    res_fomc = classifier.classify("Minutes of the Federal Open Market Committee, September 15-16, 2026.")
    assert res_fomc.category == "Macroeconomic"
    assert any("fomc" in k or "federal open market committee" in k for k in res_fomc.matched_keywords)

    # Banking supervision and financial stability
    res_stab = classifier.classify("ECB published biannual financial stability report covering banking supervision.")
    assert res_stab.category == "Macroeconomic"

    # Accounting fraud and covenant violations under Credit Event
    res_fraud = classifier.classify("Corporate borrower entered debt distress following accounting fraud and covenant violations.")
    assert res_fraud.category == "Credit Event"

    # SEC administrative outreach should remain 'Other' and not be misclassified as Geopolitical
    res_other = classifier.classify("SEC to Host Virtual National Compliance Outreach Seminar for Investment Advisers.")
    assert res_other.category == "Other"
