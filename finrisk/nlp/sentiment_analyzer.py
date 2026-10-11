"""Financial sentiment analysis module.

Supports pretrained transformer models (specifically ProsusAI/finbert) with
deterministic probability mapping to continuous risk-oriented sentiment scores.
Includes a mock-injectable predictor interface for fully offline, deterministic testing.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Supported labels in ProsusAI/finbert
FINBERT_LABELS = ("positive", "negative", "neutral")


@dataclass
class SentimentResult:
    """Structured output of financial sentiment analysis."""

    score: float  # Continuous sentiment in [-1.0, 1.0]: P(positive) - P(negative)
    label: str  # Predicted discrete class: 'positive', 'negative', or 'neutral'
    confidence: float  # Probability of the winning class in [0.0, 1.0]
    probabilities: Dict[str, float]  # Probability distribution across all classes
    model_name: str  # Identifier of the model or predictor used
    is_mock: bool = False  # Explicit flag indicating whether an injected mock predictor was used
    inference_source: str = "pretrained_transformers"  # 'pretrained_transformers', 'offline_injected_mock', or 'offline_fallback_baseline'

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return asdict(self)


def compute_continuous_score(probabilities: Dict[str, float]) -> float:
    """Map class probabilities into a continuous sentiment score in [-1.0, 1.0].

    Formula:
        score = P(positive) - P(negative)

    Properties:
        - Strongly positive (e.g., P(pos)=0.95, P(neg)=0.01) -> ~ +0.94
        - Strongly negative (e.g., P(pos)=0.02, P(neg)=0.92) -> ~ -0.90
        - Neutral dominated (e.g., P(neu)=0.90, P(pos)=0.06, P(neg)=0.04) -> ~ +0.02 (near 0)
        - Conflicted mixed sentiment (e.g., P(pos)=0.45, P(neg)=0.45) -> 0.00

    Args:
        probabilities: Dictionary containing 'positive', 'negative', and optionally 'neutral' keys.

    Returns:
        Float clamped strictly within [-1.0, 1.0], rounded to 4 decimal places.
    """
    p_pos = float(probabilities.get("positive", 0.0))
    p_neg = float(probabilities.get("negative", 0.0))

    score = p_pos - p_neg
    # Clamp to ensure numerical stability within [-1.0, 1.0]
    clamped = max(-1.0, min(1.0, score))
    return round(clamped, 4)


def deterministic_baseline_sentiment(text: str) -> Dict[str, float]:
    """Deterministic rule-based keyword sentiment baseline when FinBERT is unavailable.

    Provides stable, offline financial sentiment classification using domain stems.
    Never fabricates model weights or claims to be real FinBERT output.
    """
    lower = text.lower()
    negative_stems = (
        "fine", "penalty", "penalties", "charge", "charges", "charged", "fraud", "defraud",
        "defrauding", "cease", "desist", "investigation", "recession", "inflation", "hike",
        "cut", "cuts", "default", "defaults", "loss", "losses", "downgrade", "downgrades",
        "bankrupt", "bankruptcy", "distress", "lawsuit", "sanction", "sanctions"
    )
    positive_stems = (
        "expand", "expanded", "growth", "growing", "grow", "solid", "rebound", "boost",
        "boosted", "resilient", "surplus", "profit", "profitable", "gain", "gains",
        "upgrade", "upgrades", "recovery", "approve", "approved", "approval"
    )
    neg_count = sum(1 for stem in negative_stems if stem in lower)
    pos_count = sum(1 for stem in positive_stems if stem in lower)

    if neg_count > pos_count:
        p_neg = min(0.92, 0.60 + 0.08 * neg_count)
        p_pos = 0.05
        p_neu = round(1.0 - (p_neg + p_pos), 4)
        return {"positive": p_pos, "negative": p_neg, "neutral": p_neu}
    elif pos_count > neg_count:
        p_pos = min(0.92, 0.60 + 0.08 * pos_count)
        p_neg = 0.05
        p_neu = round(1.0 - (p_pos + p_neg), 4)
        return {"positive": p_pos, "negative": p_neg, "neutral": p_neu}
    else:
        return {"positive": 0.10, "negative": 0.10, "neutral": 0.80}


class FinBERTSentimentAnalyzer:
    """Financial sentiment analyzer targeting ProsusAI/finbert.

    Can operate in three modes:
    1. Real inference mode: Loads Hugging Face transformers pipeline if installed
       and weights are cached locally.
    2. Injected predictor mode: Uses a user-provided callable `(text) -> Dict[str, float]`
       for deterministic offline unit and integration tests without network access
       or model downloads.
    3. Deterministic fallback mode: Activates transparently when optional dependencies
       (torch, transformers) are absent, explicitly reporting `offline_fallback_baseline`.

    Note: This class does NOT silently substitute rule-based heuristics as real FinBERT;
    it explicitly signals the active predictor, fallback status, and model identity.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        predictor: Optional[Callable[[str], Dict[str, float]]] = None,
        device: Optional[str] = None,
        allow_offline_fallback: bool = True,
    ) -> None:
        """Initialize the analyzer.

        Args:
            model_name: Hugging Face model identifier (default: "ProsusAI/finbert").
            predictor: Optional injected function returning `{'positive': float, 'negative': float, 'neutral': float}`.
                       Required for offline deterministic testing without model downloads.
            device: Optional torch device string ('cpu', 'cuda').
            allow_offline_fallback: If True, automatically falls back to deterministic domain baseline
                                   when torch/transformers or model weights are missing.
        """
        self.model_name = model_name
        self.device = device
        self._predictor = predictor
        self._injected_predictor_used = predictor is not None
        self._hf_pipeline = None
        self.allow_offline_fallback = allow_offline_fallback
        self._fallback_active = False

    def _ensure_predictor(self) -> None:
        """Lazy-load the transformers model pipeline if no predictor was injected."""
        if self._predictor is not None or self._hf_pipeline is not None:
            return

        try:
            import os
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            # Safeguard Windows environments with constrained C: drive
            if os.path.exists(r"E:\pip_temp"):
                os.environ["TEMP"] = r"E:\pip_temp"
                os.environ["TMP"] = r"E:\pip_temp"
            if os.path.exists(r"E:\hf_home"):
                os.environ["HF_HOME"] = r"E:\hf_home"
                os.environ["HF_HUB_CACHE"] = r"E:\hf_home\hub"
                os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_home"
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

            is_offline = (
                os.environ.get("HF_HUB_OFFLINE") == "1"
                or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
                or os.path.exists(r"E:\hf_home")
            )

            logger.info("Loading pretrained model '%s' (offline=%s)...", self.model_name, is_offline)
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=is_offline)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, local_files_only=is_offline, use_safetensors=True
                )
            except Exception:
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name, local_files_only=is_offline
                    )
                except Exception:
                    model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            self._hf_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,  # Return all class probabilities
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
            )
        except ImportError as err:
            if self.allow_offline_fallback:
                logger.warning(
                    "FinBERT dependencies (torch/transformers) unavailable (%s); using deterministic offline fallback baseline.",
                    err,
                )
                self._fallback_active = True
                self._predictor = deterministic_baseline_sentiment
                return
            raise ImportError(
                "transformers and torch are required to run FinBERT inference. "
                "Install them or inject a deterministic mock predictor for offline testing."
            ) from err
        except Exception as err:
            if self.allow_offline_fallback:
                logger.warning(
                    "Failed to load weights for '%s' (%s); using deterministic offline fallback baseline.",
                    self.model_name,
                    err,
                )
                self._fallback_active = True
                self._predictor = deterministic_baseline_sentiment
                return
            raise RuntimeError(
                f"Failed to load weights for '{self.model_name}'. "
                f"Ensure model weights are cached locally or network access is available: {err}"
            ) from err

    def _predict_probabilities(self, text: str) -> Dict[str, float]:
        """Obtain class probabilities from injected predictor, fallback baseline, or real model."""
        if self._predictor is not None:
            raw_probs = self._predictor(text)
            # Ensure keys are lowercase and values are non-negative floats
            probs = {
                k.lower(): max(0.0, float(v))
                for k, v in raw_probs.items()
                if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit())
            }
            # Normalize sum if needed
            total = sum(probs.get(k, 0.0) for k in FINBERT_LABELS)
            if total > 0:
                return {k: round(probs.get(k, 0.0) / total, 4) for k in FINBERT_LABELS}
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        self._ensure_predictor()

        if self._predictor is not None:
            # Fallback baseline was activated in _ensure_predictor
            return self._predict_probabilities(text)

        # Run transformers pipeline
        assert self._hf_pipeline is not None
        output = self._hf_pipeline(text, truncation=True, max_length=512)

        # Output format: [[{'label': 'positive', 'score': 0.8}, ...]]
        if isinstance(output, list) and output and isinstance(output[0], list):
            items = output[0]
        elif isinstance(output, list):
            items = output
        else:
            items = []

        probs = {item["label"].lower(): float(item["score"]) for item in items if "label" in item}
        return {
            "positive": round(probs.get("positive", 0.0), 4),
            "negative": round(probs.get("negative", 0.0), 4),
            "neutral": round(probs.get("neutral", 0.0), 4),
        }

    def analyze(self, text: str) -> SentimentResult:
        """Analyze financial sentiment for a given input text.

        Args:
            text: Input financial text string.

        Returns:
            SentimentResult containing continuous score, label, confidence, and distribution.

        Raises:
            ValueError: If text is empty or whitespace-only.
        """
        if not text or not str(text).strip():
            raise ValueError("Input text cannot be empty or whitespace-only for sentiment analysis.")

        clean_text = str(text).strip()
        probs = self._predict_probabilities(clean_text)

        # Determine winning class and confidence
        best_label = max(probs, key=lambda k: probs[k])
        confidence = probs[best_label]

        # Compute continuous score: P(pos) - P(neg)
        score = compute_continuous_score(probs)

        if self._injected_predictor_used:
            is_mock = True
            inference_source = "offline_injected_mock"
            model_name = f"{self.model_name} [Injected Predictor]"
        elif self._fallback_active:
            is_mock = True
            inference_source = "offline_fallback_baseline"
            model_name = f"{self.model_name} [Offline Fallback Baseline]"
        else:
            is_mock = False
            inference_source = "pretrained_transformers"
            model_name = self.model_name

        return SentimentResult(
            score=score,
            label=best_label,
            confidence=confidence,
            probabilities=probs,
            model_name=model_name,
            is_mock=is_mock,
            inference_source=inference_source,
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze financial sentiment for a batch of texts."""
        return [self.analyze(t) for t in texts]
