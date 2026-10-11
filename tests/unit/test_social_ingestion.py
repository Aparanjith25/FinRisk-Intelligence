"""Unit tests for social media ingestion module."""

from __future__ import annotations

import pytest

from finrisk.ingestion.social_ingestion import ingest_social_csv


def test_ingest_social_valid_csv(tmp_path):
    csv_file = tmp_path / "posts.csv"
    csv_file.write_text(
        "text,source,published_at,company,url,is_synthetic\n"
        '"Retail investor sentiment bullish on EV sector",StockForum,2024-10-10T09:00:00Z,EVCorp,https://example.com/post/1,false\n'
        '"Short sellers increasing borrow rates on regional banks",FinTwit,2024-10-10T10:00:00Z,BankX,https://example.com/post/2,true\n',
        encoding="utf-8",
    )

    records = ingest_social_csv(csv_file)
    assert len(records) == 2

    real = records[0]
    assert real["text"] == "Retail investor sentiment bullish on EV sector"
    assert real["source"] == "StockForum"
    assert real["source_type"] == "social"
    assert real["company"] == "EVCorp"

    synthetic = records[1]
    assert synthetic["text"] == "Short sellers increasing borrow rates on regional banks"
    assert "Synthetic" in synthetic["source"]
    assert synthetic["source_type"] == "social_synthetic"


def test_ingest_social_column_aliases(tmp_path):
    csv_file = tmp_path / "aliases.csv"
    csv_file.write_text(
        "post_text,platform,created_at,ticker,link\n"
        '"Options volume surging on earnings release",TradeDesk,2024-10-10 14:00:00,XYZ,https://example.com/post/3\n',
        encoding="utf-8",
    )

    records = ingest_social_csv(csv_file)
    assert len(records) == 1
    rec = records[0]
    assert rec["text"] == "Options volume surging on earnings release"
    assert rec["source"] == "TradeDesk"
    assert rec["published_at"] == "2024-10-10 14:00:00"
    assert rec["company"] == "XYZ"
    assert rec["url"] == "https://example.com/post/3"


def test_ingest_social_missing_text_column_raises(tmp_path):
    csv_file = tmp_path / "missing_text.csv"
    csv_file.write_text("author,timestamp,likes\nAlice,2024-10-10,42\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing a recognized text column"):
        ingest_social_csv(csv_file)


def test_ingest_social_file_not_found(tmp_path):
    missing_file = tmp_path / "nonexistent.csv"
    with pytest.raises(FileNotFoundError):
        ingest_social_csv(missing_file)
