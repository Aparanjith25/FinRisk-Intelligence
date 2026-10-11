"""Unit tests for the data normalizer module."""

from __future__ import annotations

import pytest

from finrisk.ingestion.normalizer import (
    generate_record_id,
    normalize_record,
    normalize_records,
    parse_timestamp,
)


def test_valid_record_normalization():
    raw = {
        "text": " Federal Reserve maintains target interest rate at 5.25%.  ",
        "source": " Financial News ",
        "published_at": "2024-10-10T12:00:00Z",
        "url": " https://example.com/fed-rate ",
        "company": " Fed ",
        "source_type": "NEWS",
    }
    norm, err = normalize_record(raw)
    assert err is None
    assert norm is not None
    assert norm["text"] == "Federal Reserve maintains target interest rate at 5.25%."
    assert norm["source"] == "Financial News"
    assert norm["source_type"] == "news"
    assert norm["url"] == "https://example.com/fed-rate"
    assert norm["company"] == "Fed"
    assert norm["published_at"] == "2024-10-10T12:00:00+00:00"
    assert len(norm["record_id"]) == 16


def test_missing_and_empty_text():
    # Missing text key
    norm, err = normalize_record({"source": "Bloomberg", "source_type": "news"})
    assert norm is None
    assert "Missing required field: 'text'" in err

    # None text
    norm, err = normalize_record({"text": None, "source": "Bloomberg", "source_type": "news"})
    assert norm is None
    assert "Missing required field: 'text'" in err

    # Whitespace-only text
    norm, err = normalize_record({"text": "   \n\t  ", "source": "Bloomberg", "source_type": "news"})
    assert norm is None
    assert "Empty or whitespace-only" in err


def test_missing_source_and_source_type():
    norm, err = normalize_record({"text": "Quarterly earnings released.", "source_type": "news"})
    assert norm is None
    assert "Missing or empty required field: 'source'" in err

    norm, err = normalize_record({"text": "Quarterly earnings released.", "source": "Reuters"})
    assert norm is None
    assert "Missing or empty required field: 'source_type'" in err


def test_no_fabrication_of_optional_fields():
    raw = {
        "text": "Crude oil futures settle higher.",
        "source": "MarketWatch",
        "source_type": "news",
    }
    norm, err = normalize_record(raw)
    assert err is None
    assert norm is not None
    assert norm["published_at"] is None
    assert norm["company"] is None
    assert norm["url"] is None


def test_parse_timestamp_formats():
    # ISO format with Z
    ts1 = parse_timestamp("2024-10-10T15:30:00Z")
    assert ts1 == "2024-10-10T15:30:00+00:00"

    # RFC 2822 format (common in RSS)
    ts2 = parse_timestamp("Thu, 10 Oct 2024 15:30:00 GMT")
    assert ts2 is not None
    assert "2024-10-10" in ts2

    # Standard YYYY-MM-DD
    ts3 = parse_timestamp("2024-10-10")
    assert ts3 == "2024-10-10T00:00:00"

    # Empty string or None
    assert parse_timestamp(None) is None
    assert parse_timestamp("   ") is None


def test_parse_timestamp_invalid():
    with pytest.raises(ValueError, match="Unable to parse timestamp"):
        parse_timestamp("not-a-valid-timestamp-string")


def test_invalid_timestamp_in_record():
    raw = {
        "text": "Treasury yields edge lower.",
        "source": "CNBC",
        "source_type": "news",
        "published_at": "invalid_date_abc",
    }
    norm, err = normalize_record(raw)
    assert norm is None
    assert "Invalid timestamp" in err


def test_deduplication():
    records = [
        {
            "text": "Inflation rate slows to 2.1%.",
            "source": "BLS",
            "source_type": "news",
            "published_at": "2024-10-10T10:00:00Z",
        },
        {
            "text": "Inflation rate slows to 2.1%.",
            "source": "BLS",
            "source_type": "news",
            "published_at": "2024-10-10T10:00:00Z",
        },
        {
            "text": "Unemployment rate remains steady at 4.1%.",
            "source": "BLS",
            "source_type": "news",
            "published_at": "2024-10-10T10:00:00Z",
        },
    ]

    result = normalize_records(records, deduplicate=True)
    assert result.input_count == 3
    assert result.output_count == 2
    assert result.duplicates_removed == 1
    assert result.rejected_count == 0


def test_non_dict_rejection():
    norm, err = normalize_record("string instead of dict")  # type: ignore
    assert norm is None
    assert "not a dictionary" in err
