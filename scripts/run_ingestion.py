"""Runner script for Milestone 1: Financial Text Data Ingestion.

Loads sample RSS news and social media datasets, normalizes them into a common
schema, deduplicates, reports rejected items with reasons, and saves processed records.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finrisk.ingestion.news_ingestion import ingest_rss_file
from finrisk.ingestion.normalizer import SCHEMA_FIELDS, normalize_records
from finrisk.ingestion.social_ingestion import ingest_social_csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_ingestion")


def run_pipeline(
    rss_path: Path,
    social_path: Path,
    output_path: Path,
) -> dict:
    """Run ingestion for both sample sources and save combined records."""
    print("=" * 60)
    print("FINRISK INTELLIGENCE: INGESTION PIPELINE (MILESTONE 1)")
    print("=" * 60)

    # 1. Ingest News RSS
    print(f"\n[1] Loading RSS News feed from: {rss_path}")
    news_records = ingest_rss_file(rss_path, default_source="Financial RSS Feed")
    print(f"    Loaded raw news records: {len(news_records)}")

    # 2. Ingest Social CSV
    print(f"\n[2] Loading Social Media posts from: {social_path}")
    social_records = ingest_social_csv(social_path, default_source="Social Stream", default_synthetic=True)
    print(f"    Loaded raw social records: {len(social_records)}")

    # Combine all raw records
    raw_combined = news_records + social_records
    total_input = len(raw_combined)

    # 3. Normalize and validate
    print(f"\n[3] Normalizing and validating {total_input} total input records...")
    norm_result = normalize_records(raw_combined, deduplicate=True)

    # Log rejected records details
    if norm_result.rejected_records:
        print("\n    Rejected Records Details:")
        for rej in norm_result.rejected_records:
            print(f"      - Index {rej['index']}: Reason: {rej['reason']}")
            print(f"        Raw: {rej['record']}")

    # 4. Save to processed CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA_FIELDS)
        writer.writeheader()
        writer.writerows(norm_result.accepted_records)

    print(f"\n[4] Saved clean records to: {output_path}")

    # Summary report
    counts = {
        "input_records": total_input,
        "invalid_records": norm_result.rejected_count,
        "duplicates_removed": norm_result.duplicates_removed,
        "output_records": norm_result.output_count,
    }

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY COUNTS")
    print("=" * 60)
    print(f"  Input Records:       {counts['input_records']}")
    print(f"  Invalid Records:     {counts['invalid_records']}")
    print(f"  Duplicates Removed:  {counts['duplicates_removed']}")
    print(f"  Output Records:      {counts['output_records']}")
    print("=" * 60)

    return counts


if __name__ == "__main__":
    rss_file = PROJECT_ROOT / "data" / "sample" / "sample_news_rss.xml"
    social_file = PROJECT_ROOT / "data" / "sample" / "sample_social_posts.csv"
    out_file = PROJECT_ROOT / "data" / "processed" / "ingested_records.csv"

    run_pipeline(rss_path=rss_file, social_path=social_file, output_path=out_file)
