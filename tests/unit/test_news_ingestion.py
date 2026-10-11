"""Unit tests for RSS news ingestion module."""

from __future__ import annotations

import pytest

from finrisk.ingestion.news_ingestion import (
    fetch_rss_feed,
    ingest_rss_file,
    parse_rss_xml,
)

SAMPLE_RSS_2_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Global Market Monitor</title>
    <item>
      <title>Semiconductor sector rallies on strong forecast</title>
      <description>Leading chipmakers reported strong demand across data center units.</description>
      <link>https://example.com/chips-rally</link>
      <pubDate>Thu, 10 Oct 2024 14:00:00 GMT</pubDate>
      <source>Market Wire</source>
    </item>
    <item>
      <title>Central bank maintains benchmark policy rate</title>
      <description></description>
      <link>https://example.com/central-bank-rate</link>
      <pubDate>Thu, 10 Oct 2024 12:30:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Corporate Earnings Feed</title>
  <entry>
    <title>Q3 revenue expands 12% year over year</title>
    <summary>Automotive manufacturer exceeds quarterly revenue projections.</summary>
    <link href="https://example.com/earnings-report"/>
    <updated>2024-10-10T11:00:00Z</updated>
  </entry>
</feed>
"""


def test_parse_rss_2_xml():
    records = parse_rss_xml(SAMPLE_RSS_2_XML)
    assert len(records) == 2

    first = records[0]
    assert "Semiconductor sector rallies" in first["text"]
    assert "Leading chipmakers" in first["text"]
    assert first["source"] == "Market Wire"
    assert first["url"] == "https://example.com/chips-rally"
    assert first["published_at"] == "Thu, 10 Oct 2024 14:00:00 GMT"
    assert first["source_type"] == "news"

    # Second item with empty description falls back to channel title as source
    second = records[1]
    assert second["text"] == "Central bank maintains benchmark policy rate"
    assert second["source"] == "Global Market Monitor"


def test_parse_atom_xml():
    records = parse_rss_xml(SAMPLE_ATOM_XML)
    assert len(records) == 1
    entry = records[0]
    assert "Q3 revenue expands 12%" in entry["text"]
    assert "Automotive manufacturer" in entry["text"]
    assert entry["url"] == "https://example.com/earnings-report"
    assert entry["published_at"] == "2024-10-10T11:00:00Z"
    assert entry["source_type"] == "news"


def test_parse_empty_content():
    assert parse_rss_xml("") == []
    assert parse_rss_xml(b"   ") == []


def test_parse_malformed_xml():
    with pytest.raises(ValueError, match="Malformed XML"):
        parse_rss_xml("<unclosed>xml content")


def test_ingest_rss_file_not_found(tmp_path):
    missing_file = tmp_path / "nonexistent.xml"
    with pytest.raises(FileNotFoundError):
        ingest_rss_file(missing_file)


def test_fetch_rss_invalid_url():
    with pytest.raises(ValueError, match="Invalid URL"):
        fetch_rss_feed("ftp://invalid-protocol.com/rss")

    with pytest.raises(ValueError, match="Invalid URL"):
        fetch_rss_feed("")
