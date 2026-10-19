"""Source-Aware Financial Sentiment Router (Isolated Prototype).

Architecture:
- Routes incoming financial text to specialized model engines based strictly on
  trustworthy, normalized source metadata.
- Regulatory & Central Bank text -> Pretrained ProsusAI/finbert (Model A)
- Corporate Earnings & Financial Statement text -> Fine-Tuned Checkpoint (Model C)
- Unknown, Ambiguous, Empty, or Unlisted Sources -> Fallback to Pretrained ProsusAI/finbert (Safe Default)

Invariants:
- Never routes based on predicted sentiment, keywords, or event severity.
- Preserves the exact SentimentResult schema and P(positive) - P(negative) formula.
- Completely isolated from production codebase.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from finrisk.nlp.sentiment_analyzer import SentimentResult, compute_continuous_score

logger = logging.getLogger(__name__)

# Normalized Source Sets
REGULATORY_SOURCES: Set[str] = {
    "federal_reserve",
    "u.s. federal reserve press releases",
    "us federal reserve",
    "fed",
    "fomc",
    "sec_press_releases",
    "sec_edgar",
    "u.s. sec press releases",
    "us sec",
    "sec",
    "ecb_press_releases",
    "european central bank press releases",
    "ecb",
}

CORPORATE_SOURCES: Set[str] = {
    "corporate_news",
    "company_press",
    "earnings_release",
    "financial_phrasebank",
    "phrasebank",
    "pr_newswire",
    "businesswire",
    "investor_relations",
    "quarterly_report",
}


def normalize_source_identifier(source: Optional[str]) -> str:
    """Normalize source string for robust, case-insensitive metadata matching."""
    if not source or not str(source).strip():
        return ""
    s = str(source).strip().lower()
    # Remove excessive punctuation
    s = re.sub(r"[_\-]+", " ", s)
    s = " ".join(s.split())
    return s


@dataclass
class RoutedSentimentResult(SentimentResult):
    """Extended SentimentResult recording the routing decision provenance."""

    routed_engine: str = "pretrained_finbert"
    source_category: str = "fallback_unknown"
    normalized_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SourceAwareSentimentRouter:
    """Isolated Prototype Router dispatching text based on source provenance."""

    def __init__(
        self,
        pretrained_model_name: str = "ProsusAI/finbert",
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pretrained_name = pretrained_model_name
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            PROJECT_ROOT, "experiments", "finbert_finetuning", "best_checkpoint"
        )

        logger.info(f"Initializing SourceAwareSentimentRouter on {self.device}...")

        # 1. Load Model A (Pretrained FinBERT for Regulatory & Fallback)
        logger.info(f"Loading Regulatory/Fallback Model: {self.pretrained_name}")
        self.tok_a = AutoTokenizer.from_pretrained(self.pretrained_name)
        self.mod_a = AutoModelForSequenceClassification.from_pretrained(self.pretrained_name)
        self.mod_a.to(self.device)
        self.mod_a.eval()

        # 2. Load Model C (Fine-Tuned FinBERT for Corporate Earnings)
        logger.info(f"Loading Corporate Specialist Checkpoint: {self.checkpoint_dir}")
        self.tok_c = AutoTokenizer.from_pretrained(self.checkpoint_dir)
        self.mod_c = AutoModelForSequenceClassification.from_pretrained(self.checkpoint_dir)
        self.mod_c.to(self.device)
        self.mod_c.eval()

        self.id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def route_source(self, source: Optional[str]) -> Tuple[str, str]:
        """Determine target model engine from source metadata.

        Returns:
            Tuple of (engine_name, source_category)
            engine_name in {'regulatory_model', 'corporate_model', 'fallback_model'}
        """
        norm_source = normalize_source_identifier(source)
        if not norm_source:
            return "fallback_model", "empty_or_missing_source"

        # Check Regulatory
        for reg_s in REGULATORY_SOURCES:
            reg_norm = normalize_source_identifier(reg_s)
            if norm_source == reg_norm or reg_norm in norm_source:
                return "regulatory_model", "regulatory_official"

        # Check Corporate
        for corp_s in CORPORATE_SOURCES:
            corp_norm = normalize_source_identifier(corp_s)
            if norm_source == corp_norm or corp_norm in norm_source:
                return "corporate_model", "corporate_financial"

        # Unknown / Unlisted -> Fallback to robust baseline
        return "fallback_model", "unknown_source_fallback"

    def analyze(self, text: str, source: Optional[str] = None) -> RoutedSentimentResult:
        """Route and analyze financial text while strictly maintaining SentimentResult schema."""
        if not text or not str(text).strip():
            # Graceful neutral contract
            return RoutedSentimentResult(
                score=0.0,
                label="neutral",
                confidence=0.3333,
                probabilities={"positive": 0.3333, "negative": 0.3333, "neutral": 0.3334},
                model_name="source_router_empty_handler",
                inference_source="source_aware_router",
                routed_engine="none",
                source_category="empty_input",
                normalized_source=normalize_source_identifier(source),
            )

        engine_choice, source_category = self.route_source(source)

        if engine_choice == "corporate_model":
            tok = self.tok_c
            mod = self.mod_c
            engine_label = "finbert_corporate_checkpoint"
        else:
            # Both 'regulatory_model' and 'fallback_model' use Model A
            tok = self.tok_a
            mod = self.mod_a
            engine_label = "prosus_finbert_baseline"

        inputs = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        ).to(self.device)

        with torch.no_grad():
            logits = mod(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        p_pos = float(probs[0])
        p_neg = float(probs[1])
        p_neu = float(probs[2])

        score = max(-1.0, min(1.0, p_pos - p_neg))
        prob_dict = {"positive": round(p_pos, 4), "negative": round(p_neg, 4), "neutral": round(p_neu, 4)}
        best_label = max(prob_dict, key=prob_dict.get)

        return RoutedSentimentResult(
            score=round(score, 4),
            label=best_label,
            confidence=round(prob_dict[best_label], 4),
            probabilities=prob_dict,
            model_name=f"{engine_label} ({engine_choice})",
            inference_source="source_aware_router",
            routed_engine=engine_choice,
            source_category=source_category,
            normalized_source=normalize_source_identifier(source),
        )
