"""Data normalization module for financial text ingestion.

Normalizes records from disparate sources (news RSS, social media CSV, etc.)
into a standard, validated schema for downstream processing.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Standard target schema fields
REQUIRED_FIELDS = ("text", "source", "source_type")
SCHEMA_FIELDS = (
    "record_id",
    "source",
    "published_at",
    "text",
    "url",
    "company",
    "source_type",
)
PUBLIC_NEWS_SCHEMA_FIELDS = SCHEMA_FIELDS + (
    "provenance",
    "retrieved_at",
)


@dataclass
class NormalizationResult:
    """Summary of batch normalization results."""

    accepted_records: List[Dict[str, Any]] = field(default_factory=list)
    rejected_records: List[Dict[str, Any]] = field(default_factory=list)
    duplicates_removed: int = 0

    @property
    def input_count(self) -> int:
        return len(self.accepted_records) + len(self.rejected_records) + self.duplicates_removed

    @property
    def output_count(self) -> int:
        return len(self.accepted_records)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_records)


def parse_timestamp(value: Optional[str]) -> Optional[str]:
    """Parse various datetime representations into ISO 8601 format.

    Supports RFC 2822 (RSS format) and ISO 8601 formats.
    Returns None if value is None or empty. Does not fabricate dates.
    Raises ValueError if a provided string cannot be parsed.
    """
    if not value or not str(value).strip():
        return None

    val_str = str(value).strip()

    # Try ISO 8601 parsing first
    try:
        # Handle trailing Z
        iso_str = val_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_str)
        return dt.isoformat()
    except (ValueError, TypeError):
        pass

    # Try RFC 2822 / 822 (common in RSS feeds, e.g. "Mon, 02 Jan 2023 15:04:05 GMT")
    try:
        dt = parsedate_to_datetime(val_str)
        return dt.isoformat()
    except (ValueError, TypeError, IndexError):
        pass

    # Common date formats without timezone
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(val_str, fmt)
            return dt.isoformat()
        except ValueError:
            continue

    raise ValueError(f"Unable to parse timestamp: '{val_str}'")


def generate_record_id(source: str, text: str, published_at: Optional[str] = None) -> str:
    """Deterministically generate a unique 16-character hex record ID."""
    content = f"{source}|{text}|{published_at or ''}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def normalize_record(raw: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate and normalize a single record into the target schema.

    Returns:
        (normalized_dict, None) if valid.
        (None, rejection_reason) if invalid.
    """
    if not isinstance(raw, dict):
        return None, "Record is not a dictionary"

    # Validate and clean text
    raw_text = raw.get("text")
    if raw_text is None:
        return None, "Missing required field: 'text'"
    text = str(raw_text).strip()
    if not text:
        return None, "Empty or whitespace-only 'text' field"

    # Validate source
    raw_source = raw.get("source")
    if not raw_source or not str(raw_source).strip():
        return None, "Missing or empty required field: 'source'"
    source = str(raw_source).strip()

    # Validate source_type
    raw_source_type = raw.get("source_type")
    if not raw_source_type or not str(raw_source_type).strip():
        return None, "Missing or empty required field: 'source_type'"
    source_type = str(raw_source_type).strip().lower()

    # Parse published_at safely without fabricating
    raw_published_at = raw.get("published_at")
    published_at: Optional[str] = None
    if raw_published_at is not None and str(raw_published_at).strip():
        try:
            published_at = parse_timestamp(str(raw_published_at))
        except ValueError as err:
            return None, f"Invalid timestamp: {err}"

    # Handle company safely without fabricating
    company_val = raw.get("company")
    company = str(company_val).strip() if (company_val is not None and str(company_val).strip()) else None

    # Handle url safely without fabricating
    url_val = raw.get("url")
    url = str(url_val).strip() if (url_val is not None and str(url_val).strip()) else None

    # Determine or generate record_id
    raw_id = raw.get("record_id")
    if raw_id is not None and str(raw_id).strip():
        record_id = str(raw_id).strip()
    else:
        record_id = generate_record_id(source=source, text=text, published_at=published_at)

    normalized = {
        "record_id": record_id,
        "source": source,
        "published_at": published_at,
        "text": text,
        "url": url,
        "company": company,
        "source_type": source_type,
    }

    if "provenance" in raw and raw["provenance"] is not None:
        normalized["provenance"] = str(raw["provenance"]).strip()
    if "retrieved_at" in raw and raw["retrieved_at"] is not None:
        normalized["retrieved_at"] = str(raw["retrieved_at"]).strip()

    return normalized, None


def normalize_records(
    records: Iterable[Dict[str, Any]],
    deduplicate: bool = True,
) -> NormalizationResult:
    """Normalize a batch of records, deduplicate, and record rejection details.

    Args:
        records: Iterable of raw record dictionaries.
        deduplicate: If True, remove duplicate items based on (source, text, published_at).

    Returns:
        NormalizationResult detailing accepted, rejected, and duplicate records.
    """
    result = NormalizationResult()
    seen_fingerprints: set[str] = set()

    for idx, raw in enumerate(records):
        norm, reason = normalize_record(raw)
        if norm is None:
            result.rejected_records.append({
                "index": idx,
                "record": raw,
                "reason": reason or "Unknown validation error",
            })
            logger.warning("Rejected record %d: %s", idx, reason)
            continue

        if deduplicate:
            # Fingerprint based on source, text content, and published timestamp
            fingerprint = f"{norm['source']}|{norm['text']}|{norm['published_at'] or ''}"
            if fingerprint in seen_fingerprints:
                result.duplicates_removed += 1
                logger.info("Removed duplicate record at index %d (ID %s)", idx, norm["record_id"])
                continue
            seen_fingerprints.add(fingerprint)

        result.accepted_records.append(norm)

    return result
