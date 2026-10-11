"""Integration test for the RiskEngine pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

from finrisk.nlp.event_classifier import EVENT_CATEGORIES, RuleBasedEventClassifier
from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer
from finrisk.risk_engine.engine import RISK_SIGNAL_FIELDS, RiskEngine
from scripts.run_risk_engine import run_batch_risk_engine, sample_offline_finbert_predictor


def test_risk_engine_integration_pipeline(tmp_path):
    project_root = Path(__file__).resolve().parent.parent.parent
    ingested_csv = project_root / "data" / "processed" / "ingested_records.csv"
    output_signals_csv = tmp_path / "test_risk_signals.csv"

    assert ingested_csv.exists(), f"Ingested records CSV missing: {ingested_csv}"

    # Run batch engine using the test runner function
    signals = run_batch_risk_engine(
        input_csv=ingested_csv,
        output_csv=output_signals_csv,
    )

    # All 8 ingested records should successfully yield risk signals
    assert len(signals) == 8
    assert output_signals_csv.exists()

    # Verify CSV file contents
    with open(output_signals_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert list(reader.fieldnames) == list(RISK_SIGNAL_FIELDS)
        rows = list(reader)
        assert len(rows) == 8

        for row in rows:
            # Check fields
            assert row["record_id"], "record_id must be populated"
            assert row["text"], "text must be populated"
            assert row["source"], "source must be populated"
            assert row["event_type"] in EVENT_CATEGORIES
            assert row["sentiment_label"] in ("positive", "negative", "neutral")

            # Check numeric bounds
            score = float(row["sentiment_score"])
            assert -1.0 <= score <= 1.0

            conf = float(row["sentiment_confidence"])
            assert 0.0 <= conf <= 1.0

            impact = int(row["impact_score"])
            assert 1 <= impact <= 10

            assert row["scoring_rationale"], "scoring_rationale must explain calculation"
