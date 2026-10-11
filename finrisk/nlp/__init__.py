"""NLP module for financial text sentiment and risk disclosure analysis."""

from finrisk.nlp.event_classifier import (
    EVENT_CATEGORIES,
    BaseEventClassifier,
    EventClassificationResult,
    RuleBasedEventClassifier,
)
from finrisk.nlp.sentiment_analyzer import (
    FinBERTSentimentAnalyzer,
    SentimentResult,
    compute_continuous_score,
)

__all__ = [
    "compute_continuous_score",
    "FinBERTSentimentAnalyzer",
    "SentimentResult",
    "EVENT_CATEGORIES",
    "BaseEventClassifier",
    "EventClassificationResult",
    "RuleBasedEventClassifier",
]
