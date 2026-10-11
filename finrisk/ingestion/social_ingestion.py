"""Social media text ingestion module.

Loads financial social media records from CSV files, mapping documented
columns to standard fields and clearly distinguishing synthetic vs public records.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Supported column aliases for flexible ingestion
TEXT_COLUMNS = ("text", "post_text", "content", "tweet", "message")
SOURCE_COLUMNS = ("source", "platform")
DATE_COLUMNS = ("published_at", "created_at", "timestamp", "date")
URL_COLUMNS = ("url", "link")
COMPANY_COLUMNS = ("company", "ticker", "symbol")
SYNTHETIC_COLUMNS = ("is_synthetic", "synthetic")


def _find_column_value(row: Dict[str, str], aliases: tuple[str, ...]) -> Optional[str]:
    """Find the first matching non-empty column value among aliases."""
    for alias in aliases:
        if alias in row and row[alias] is not None:
            val = str(row[alias]).strip()
            if val:
                return val
    return None


def ingest_social_csv(
    file_path: Union[str, Path],
    default_source: str = "Social Media",
    default_synthetic: bool = False,
) -> List[Dict[str, Any]]:
    """Load social media records from a CSV file.

    Args:
        file_path: Path to the CSV file.
        default_source: Fallback source name if not provided in row.
        default_synthetic: If True, mark records as synthetic unless explicitly specified.

    Returns:
        List of raw dictionaries ready for normalization.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns cannot be found or the file is invalid.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Social CSV file not found: {path}")

    records: List[Dict[str, Any]] = []

    with open(path, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or headerless CSV file: {path}")

        # Validate that at least one recognizable text column exists
        normalized_headers = [h.strip().lower() for h in reader.fieldnames if h]
        matched_text_col = any(alias in normalized_headers for alias in TEXT_COLUMNS)
        if not matched_text_col:
            raise ValueError(
                f"CSV in '{path}' is missing a recognized text column. "
                f"Available columns: {reader.fieldnames}. "
                f"Expected one of: {TEXT_COLUMNS}"
            )

        for row_idx, raw_row in enumerate(reader, start=1):
            # Normalize row keys to lowercase stripped
            row = {k.strip().lower(): v for k, v in raw_row.items() if k is not None}

            text = _find_column_value(row, TEXT_COLUMNS)
            source_val = _find_column_value(row, SOURCE_COLUMNS) or default_source
            published_at = _find_column_value(row, DATE_COLUMNS)
            url = _find_column_value(row, URL_COLUMNS)
            company = _find_column_value(row, COMPANY_COLUMNS)

            # Determine whether the record is synthetic
            is_synthetic = default_synthetic
            synthetic_str = _find_column_value(row, SYNTHETIC_COLUMNS)
            if synthetic_str is not None:
                is_synthetic = synthetic_str.lower() in ("true", "1", "yes", "synthetic")

            # Clearly distinguish synthetic sample posts from real public-source records
            if is_synthetic:
                effective_source = f"{source_val} (Synthetic)"
                source_type = "social_synthetic"
            else:
                effective_source = source_val
                source_type = "social"

            records.append({
                "text": text if text is not None else "",
                "source": effective_source,
                "published_at": published_at,
                "url": url,
                "company": company,
                "source_type": source_type,
            })

    logger.info("Loaded %d social records from %s", len(records), path)
    return records
