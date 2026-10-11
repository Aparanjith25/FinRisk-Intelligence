"""Public financial news ingestion module.

Ingests and normalizes real, publicly accessible financial news feeds (e.g. U.S. Federal
Reserve, SEC, ECB) while recording data provenance and ensuring fault tolerance across feeds.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finrisk.ingestion.news_ingestion import parse_rss_xml
from finrisk.ingestion.normalizer import PUBLIC_NEWS_SCHEMA_FIELDS, normalize_records

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "public_feeds.yaml"
DEFAULT_USER_AGENT = "FinRisk-Intelligence/1.0 (+https://github.com/finrisk-intelligence)"


@dataclass
class FeedConfig:
    """Configuration definition for a public financial news feed."""

    feed_id: str
    name: str
    url: str
    description: str
    default_category: str = "Macroeconomic"
    provenance: str = "real_public_news"
    enabled: bool = True

    @classmethod
    def from_dict(cls, feed_id: str, data: Dict[str, Any]) -> FeedConfig:
        return cls(
            feed_id=feed_id,
            name=data.get("name", feed_id),
            url=data.get("url", ""),
            description=data.get("description", ""),
            default_category=data.get("default_category", "Macroeconomic"),
            provenance=data.get("provenance", "real_public_news"),
            enabled=data.get("enabled", True),
        )


@dataclass
class FeedFetchStatus:
    """Operational status report for a single feed retrieval attempt."""

    feed_id: str
    name: str
    url: str
    success: bool
    status_code: Optional[int] = None
    article_count: int = 0
    error_message: Optional[str] = None
    retrieved_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PublicNewsIngestionResult:
    """Consolidated results of a public news ingestion run."""

    feed_statuses: List[FeedFetchStatus] = field(default_factory=list)
    accepted_records: List[Dict[str, Any]] = field(default_factory=list)
    rejected_records: List[Dict[str, Any]] = field(default_factory=list)
    duplicates_removed: int = 0
    total_raw_count: int = 0
    output_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feed_statuses": [s.to_dict() for s in self.feed_statuses],
            "total_raw_count": self.total_raw_count,
            "output_count": self.output_count,
            "duplicates_removed": self.duplicates_removed,
            "rejected_count": len(self.rejected_records),
            "accepted_records": self.accepted_records,
        }


def load_public_feed_configs(config_path: Optional[Union[str, Path]] = None) -> List[FeedConfig]:
    """Load public feed definitions from YAML configuration file."""
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Public feeds config file not found: {path}")

    with open(path, mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "feeds" not in data:
        raise ValueError(f"Invalid public feeds configuration format in {path}: expected 'feeds' key")

    configs: List[FeedConfig] = []
    for fid, fdef in data["feeds"].items():
        configs.append(FeedConfig.from_dict(fid, fdef))

    return configs


def fetch_feed_content(
    url: str,
    timeout: int = 10,
    max_retries: int = 2,
    headers: Optional[Dict[str, str]] = None,
) -> tuple[str, int]:
    """Fetch raw XML feed content over HTTP with bounded timeouts and retries.

    Returns:
        tuple of (raw_xml_text, http_status_code)

    Raises:
        ConnectionError: If all retry attempts fail.
    """
    req_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        req_headers.update(headers)

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.get(
                url,
                headers=req_headers,
                timeout=float(timeout),
                follow_redirects=True,
            )
            if resp.status_code == 200:
                return resp.text, resp.status_code
            elif resp.status_code >= 500 and attempt < max_retries:
                # Transient server error, retry after backoff
                time.sleep(1.0 * attempt)
                continue
            else:
                raise ConnectionError(f"HTTP {resp.status_code} fetching feed: {resp.reason_phrase}")
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.0 * attempt)
                continue
        except Exception as exc:
            last_error = exc
            break

    raise ConnectionError(f"Failed to fetch feed at {url} after {max_retries} attempts: {last_error}")


def parse_public_feed(
    xml_content: Union[str, bytes],
    feed_config: FeedConfig,
    retrieved_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Parse feed XML and attach data provenance metadata."""
    retrieval_time = retrieved_at or datetime.now(timezone.utc).isoformat()
    parsed_items = parse_rss_xml(xml_content, default_source=feed_config.name)

    enriched_records: List[Dict[str, Any]] = []
    for item in parsed_items:
        record = {
            "text": item.get("text", ""),
            "source": feed_config.name,
            "source_url": feed_config.url,
            "published_at": item.get("published_at"),
            "url": item.get("url"),
            "company": None,  # Will be extracted or left None (never fabricated)
            "source_type": "public_news",
            "provenance": feed_config.provenance,
            "retrieved_at": retrieval_time,
        }
        enriched_records.append(record)

    return enriched_records


def fetch_public_news(
    config_path: Optional[Union[str, Path]] = None,
    timeout: int = 10,
    max_retries: int = 2,
    enabled_only: bool = True,
) -> PublicNewsIngestionResult:
    """Fetch all enabled public news feeds, normalize, and report per-feed statuses.

    Fault-tolerant: if one feed fails (e.g. network timeout or 404), other feeds
    are still processed successfully.
    """
    feed_configs = load_public_feed_configs(config_path)
    if enabled_only:
        feed_configs = [c for c in feed_configs if c.enabled]

    statuses: List[FeedFetchStatus] = []
    all_raw_records: List[Dict[str, Any]] = []
    retrieval_time = datetime.now(timezone.utc).isoformat()

    for cfg in feed_configs:
        logger.info("Fetching public feed '%s' from %s", cfg.name, cfg.url)
        try:
            xml_text, status_code = fetch_feed_content(
                cfg.url,
                timeout=timeout,
                max_retries=max_retries,
            )
            raw_items = parse_public_feed(xml_text, cfg, retrieved_at=retrieval_time)
            statuses.append(
                FeedFetchStatus(
                    feed_id=cfg.feed_id,
                    name=cfg.name,
                    url=cfg.url,
                    success=True,
                    status_code=status_code,
                    article_count=len(raw_items),
                    error_message=None,
                    retrieved_at=retrieval_time,
                )
            )
            all_raw_records.extend(raw_items)
            logger.info("Successfully fetched %d articles from '%s'", len(raw_items), cfg.name)
        except Exception as err:
            logger.warning("Failed to fetch feed '%s' (%s): %s", cfg.name, cfg.url, err)
            statuses.append(
                FeedFetchStatus(
                    feed_id=cfg.feed_id,
                    name=cfg.name,
                    url=cfg.url,
                    success=False,
                    status_code=None,
                    article_count=0,
                    error_message=str(err),
                    retrieved_at=retrieval_time,
                )
            )

    # Normalize and deduplicate combined raw records
    norm_result = normalize_records(all_raw_records, deduplicate=True)

    return PublicNewsIngestionResult(
        feed_statuses=statuses,
        accepted_records=norm_result.accepted_records,
        rejected_records=norm_result.rejected_records,
        duplicates_removed=norm_result.duplicates_removed,
        total_raw_count=len(all_raw_records),
        output_count=norm_result.output_count,
    )


def save_public_news_csv(
    records: List[Dict[str, Any]],
    output_path: Union[str, Path],
    merge_existing: bool = True,
) -> Path:
    """Save normalized public news records to CSV file using public schema, deduplicating by record_id."""
    out = Path(output_path)
    all_records = list(records)
    if merge_existing and out.exists():
        try:
            with open(out, mode="r", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
                all_records = existing + all_records
        except Exception:
            pass

    # Deduplicate deterministically by record_id
    seen_ids = set()
    deduped = []
    for r in all_records:
        rid = r.get("record_id")
        if rid and rid in seen_ids:
            continue
        if rid:
            seen_ids.add(rid)
        deduped.append(r)

    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLIC_NEWS_SCHEMA_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deduped)

    logger.info("Saved %d public news records to %s", len(deduped), out)
    return out
