"""Integration tests for the end-to-end ingestion pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

from finrisk.ingestion.normalizer import SCHEMA_FIELDS
from scripts.run_ingestion import run_pipeline


def test_full_ingestion_pipeline_with_sample_files(tmp_path):
    # Locate actual sample datasets
    project_root = Path(__file__).resolve().parent.parent.parent
    sample_news = project_root / "data" / "sample" / "sample_news_rss.xml"
    sample_social = project_root / "data" / "sample" / "sample_social_posts.csv"
    output_file = tmp_path / "processed_test_records.csv"

    assert sample_news.exists(), f"Sample news file missing: {sample_news}"
    assert sample_social.exists(), f"Sample social file missing: {sample_social}"

    counts = run_pipeline(
        rss_path=sample_news,
        social_path=sample_social,
        output_path=output_file,
    )

    # In our sample data:
    # 4 RSS items + 6 social rows = 10 total input records
    # 1 invalid social row (whitespace only text)
    # 1 duplicate social row
    # 8 valid output records (4 news + 4 social)
    assert counts["input_records"] == 10
    assert counts["invalid_records"] == 1
    assert counts["duplicates_removed"] == 1
    assert counts["output_records"] == 8

    # Verify generated output file exists and has correct CSV structure
    assert output_file.exists()

    with open(output_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert list(reader.fieldnames) == list(SCHEMA_FIELDS)
        rows = list(reader)
        assert len(rows) == 8

        # Check required fields are populated for every row
        for row in rows:
            assert row["record_id"], "record_id must not be empty"
            assert row["source"], "source must not be empty"
            assert row["text"], "text must not be empty"
            assert row["source_type"] in ("news", "social", "social_synthetic")

        # Verify synthetic records are properly flagged in source_type
        synthetic_rows = [r for r in rows if r["source_type"] == "social_synthetic"]
        assert len(synthetic_rows) == 4
