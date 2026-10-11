"""Unit tests for the public financial news ingestion module."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from finrisk.ingestion.public_news import (
    FeedConfig,
    FeedFetchStatus,
    PublicNewsIngestionResult,
    fetch_feed_content,
    fetch_public_news,
    load_public_feed_configs,
    parse_public_feed,
    save_public_news_csv,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
FED_FIXTURE_PATH = FIXTURES_DIR / "sample_fed_feed.xml"
SEC_FIXTURE_PATH = FIXTURES_DIR / "sample_sec_feed.xml"


def test_load_public_feed_configs_default():
    """Verify loading default configs/public_feeds.yaml."""
    configs = load_public_feed_configs()
    assert len(configs) >= 2
    feed_ids = [c.feed_id for c in configs]
    assert "federal_reserve" in feed_ids
    assert "sec_press_releases" in feed_ids
    assert "ecb_press_releases" in feed_ids

    fed = next(c for c in configs if c.feed_id == "federal_reserve")
    assert "Federal Reserve" in fed.name
    assert fed.url.startswith("https://")
    assert fed.provenance == "real_public_news"
    assert fed.enabled is True


def test_load_public_feed_configs_file_not_found(tmp_path):
    """Verify FileNotFoundError when config file is missing."""
    with pytest.raises(FileNotFoundError):
        load_public_feed_configs(tmp_path / "nonexistent.yaml")


def test_load_public_feed_configs_invalid_format(tmp_path):
    """Verify ValueError when config file lacks 'feeds' key."""
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("invalid_key: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 'feeds' key"):
        load_public_feed_configs(bad_config)


def test_parse_public_feed_fed_fixture():
    """Verify parsing real public feed XML with Federal Reserve fixture."""
    xml_content = FED_FIXTURE_PATH.read_text(encoding="utf-8")
    cfg = FeedConfig(
        feed_id="fed_test",
        name="U.S. Federal Reserve",
        url="https://www.federalreserve.gov/feeds/press_all.xml",
        description="Fed press releases",
        provenance="real_public_news",
    )
    records = parse_public_feed(xml_content, cfg, retrieved_at="2026-10-10T12:00:00Z")

    assert len(records) == 2
    rec = records[0]
    assert "FOMC statement" in rec["text"]
    assert rec["source"] == "U.S. Federal Reserve"
    assert rec["source_url"] == "https://www.federalreserve.gov/feeds/press_all.xml"
    assert rec["provenance"] == "real_public_news"
    assert rec["source_type"] == "public_news"
    assert rec["retrieved_at"] == "2026-10-10T12:00:00Z"
    assert rec["url"] == "https://www.federalreserve.gov/newsevents/pressreleases/monetary20261001a.htm"
    assert rec["published_at"] is not None
    # Verify company is not fabricated
    assert rec["company"] is None


def test_parse_public_feed_sec_fixture():
    """Verify parsing SEC press releases fixture."""
    xml_content = SEC_FIXTURE_PATH.read_text(encoding="utf-8")
    cfg = FeedConfig(
        feed_id="sec_test",
        name="U.S. SEC",
        url="https://www.sec.gov/news/pressreleases.rss",
        description="SEC press releases",
        provenance="real_public_news",
    )
    records = parse_public_feed(xml_content, cfg, retrieved_at="2026-10-10T12:00:00Z")

    assert len(records) == 2
    rec = records[0]
    assert "SEC Charges" in rec["text"]
    assert rec["source"] == "U.S. SEC"
    assert rec["provenance"] == "real_public_news"
    assert rec["url"] == "https://www.sec.gov/newsroom/press-releases/2026-95-sec-charges-commercial-real-estate-issuer"


@patch("finrisk.ingestion.public_news.httpx.get")
def test_fetch_feed_content_success(mock_get):
    """Verify successful HTTP fetch returning text and status code."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<rss><channel><title>Test</title></channel></rss>"
    mock_get.return_value = mock_resp

    text, code = fetch_feed_content("https://example.com/rss.xml", timeout=5)
    assert code == 200
    assert "<rss>" in text
    mock_get.assert_called_once()


@patch("finrisk.ingestion.public_news.time.sleep")
@patch("finrisk.ingestion.public_news.httpx.get")
def test_fetch_feed_content_retry_on_500(mock_get, mock_sleep):
    """Verify retry logic on transient HTTP 500 error."""
    mock_500 = MagicMock()
    mock_500.status_code = 500
    mock_500.reason_phrase = "Internal Server Error"

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.text = "<rss><channel><title>Success</title></channel></rss>"

    # Attempt 1 returns 500, attempt 2 returns 200
    mock_get.side_effect = [mock_500, mock_200]

    text, code = fetch_feed_content("https://example.com/rss.xml", timeout=5, max_retries=2)
    assert code == 200
    assert "Success" in text
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


@patch("finrisk.ingestion.public_news.httpx.get")
def test_fetch_feed_content_failure_raises_connection_error(mock_get):
    """Verify ConnectionError raised on 404 or repeated failure."""
    mock_404 = MagicMock()
    mock_404.status_code = 404
    mock_404.reason_phrase = "Not Found"
    mock_get.return_value = mock_404

    with pytest.raises(ConnectionError, match="HTTP 404"):
        fetch_feed_content("https://example.com/missing.xml", timeout=5, max_retries=1)


@patch("finrisk.ingestion.public_news.fetch_feed_content")
def test_fetch_public_news_partial_failure_tolerance(mock_fetch, tmp_path):
    """Verify fault isolation: when one feed fails, others still succeed."""
    fed_xml = FED_FIXTURE_PATH.read_text(encoding="utf-8")

    # Feed 1 succeeds with Fed XML, Feed 2 throws ConnectionError, Feed 3 succeeds with empty XML
    def side_effect(url, **kwargs):
        if "federalreserve" in url:
            return fed_xml, 200
        elif "sec.gov" in url:
            raise ConnectionError("Network unreachable")
        else:
            return "<rss><channel><title>ECB</title></channel></rss>", 200

    mock_fetch.side_effect = side_effect

    result = fetch_public_news()
    assert isinstance(result, PublicNewsIngestionResult)
    assert len(result.feed_statuses) >= 2

    # Verify Fed succeeded
    fed_status = next(s for s in result.feed_statuses if "federal_reserve" in s.feed_id)
    assert fed_status.success is True
    assert fed_status.article_count == 2
    assert fed_status.error_message is None

    # Verify SEC failed gracefully without crashing pipeline
    sec_status = next(s for s in result.feed_statuses if "sec_press" in s.feed_id)
    assert sec_status.success is False
    assert sec_status.article_count == 0
    assert "Network unreachable" in str(sec_status.error_message)

    # Verify accepted records include Fed articles
    assert result.output_count >= 2
    assert any("FOMC statement" in r["text"] for r in result.accepted_records)


def test_save_public_news_csv(tmp_path):
    """Verify saving public news records to CSV format with correct headers."""
    sample_records = [
        {
            "record_id": "rec_001",
            "source": "Federal Reserve",
            "published_at": "2026-10-01T14:00:00Z",
            "text": "FOMC maintains federal funds rate.",
            "url": "https://www.federalreserve.gov/press/1",
            "company": None,
            "source_type": "public_news",
            "provenance": "real_public_news",
            "retrieved_at": "2026-10-10T12:00:00Z",
        }
    ]
    out_file = tmp_path / "test_public_records.csv"
    saved_path = save_public_news_csv(sample_records, out_file)
    assert saved_path.exists()

    with open(saved_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["record_id"] == "rec_001"
        assert reader[0]["source"] == "Federal Reserve"
        assert reader[0]["provenance"] == "real_public_news"
        assert reader[0]["retrieved_at"] == "2026-10-10T12:00:00Z"
