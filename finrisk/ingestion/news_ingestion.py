"""RSS news ingestion module for financial text data.

Parses RSS and Atom XML feeds using the Python standard library with robust
error handling, network timeouts, and offline file parsing support.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "FinRisk-Intelligence/1.0 (+https://github.com/finrisk-intelligence)"


def parse_rss_xml(
    xml_content: Union[str, bytes],
    default_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Parse RSS 2.0 or Atom XML content into raw record dictionaries.

    Does not make network requests; suitable for offline parsing and unit tests.

    Args:
        xml_content: XML string or bytes.
        default_source: Fallback source name if not found in feed.

    Returns:
        List of raw dictionaries ready for normalization.
    """
    if isinstance(xml_content, str):
        xml_bytes = xml_content.encode("utf-8")
    else:
        xml_bytes = xml_content

    if not xml_bytes or not xml_bytes.strip():
        logger.warning("Empty XML content received for parsing.")
        return []

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as err:
        logger.error("XML parse error: %s", err)
        raise ValueError(f"Malformed XML feed: {err}") from err

    records: List[Dict[str, Any]] = []

    # Detect feed channel title as fallback source
    channel_title: Optional[str] = None
    channel = root.find("channel")
    if channel is not None:
        title_el = channel.find("title")
        if title_el is not None and title_el.text:
            channel_title = title_el.text.strip()

    source_name = default_source or channel_title or "RSS News Feed"

    # 1. Look for RSS 2.0 items (<channel><item> or <item>)
    items = root.findall(".//item")
    if items:
        for item in items:
            title_el = item.find("title")
            desc_el = item.find("description")
            link_el = item.find("link")
            pub_date_el = item.find("pubDate")
            source_el = item.find("source")

            title = title_el.text.strip() if (title_el is not None and title_el.text) else ""
            description = desc_el.text.strip() if (desc_el is not None and desc_el.text) else ""

            # Combine title and description for comprehensive text, or use available
            if title and description:
                text = f"{title}. {description}"
            else:
                text = title or description

            url = link_el.text.strip() if (link_el is not None and link_el.text) else None
            published_at = pub_date_el.text.strip() if (pub_date_el is not None and pub_date_el.text) else None
            item_source = (
                source_el.text.strip()
                if (source_el is not None and source_el.text)
                else source_name
            )

            records.append({
                "text": text,
                "source": item_source,
                "published_at": published_at,
                "url": url,
                "source_type": "news",
            })
        return records

    # 2. Look for Atom entries (<entry>)
    # Handle possible namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    entries = root.findall(f".//{ns}entry")
    for entry in entries:
        title_el = entry.find(f"{ns}title")
        summary_el = entry.find(f"{ns}summary")
        if summary_el is None:
            summary_el = entry.find(f"{ns}content")

        link_el = entry.find(f"{ns}link")
        updated_el = entry.find(f"{ns}updated")
        if updated_el is None:
            updated_el = entry.find(f"{ns}published")

        title = title_el.text.strip() if (title_el is not None and title_el.text) else ""
        summary = summary_el.text.strip() if (summary_el is not None and summary_el.text) else ""

        if title and summary:
            text = f"{title}. {summary}"
        else:
            text = title or summary

        url = None
        if link_el is not None:
            url = link_el.attrib.get("href") or link_el.text

        published_at = updated_el.text.strip() if (updated_el is not None and updated_el.text) else None

        records.append({
            "text": text,
            "source": source_name,
            "published_at": published_at,
            "url": url,
            "source_type": "news",
        })

    return records


def fetch_rss_feed(
    url: str,
    timeout: int = 15,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """Fetch RSS XML from a public URL using standard library networking.

    Args:
        url: Public RSS feed URL.
        timeout: Network timeout in seconds (default 15s).
        headers: Optional HTTP headers dictionary.

    Returns:
        Raw XML feed text content.

    Raises:
        ValueError: If URL is invalid or protocol is unsupported.
        ConnectionError: If network connection fails or times out.
    """
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"Invalid URL: {url}. Must start with http:// or https://")

    req_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding, errors="replace")
    except urllib.error.HTTPError as err:
        logger.error("HTTP error fetching %s: code %d - %s", url, err.code, err.reason)
        raise ConnectionError(f"HTTP error {err.code} fetching RSS feed: {err.reason}") from err
    except urllib.error.URLError as err:
        logger.error("Network error fetching %s: %s", url, err.reason)
        raise ConnectionError(f"Network error fetching RSS feed: {err.reason}") from err
    except TimeoutError as err:
        logger.error("Timeout fetching %s after %d seconds", url, timeout)
        raise ConnectionError(f"Timeout fetching RSS feed after {timeout} seconds") from err


def ingest_rss_file(
    file_path: Union[str, Path],
    default_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load and parse an RSS XML feed from a local file path."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"RSS file not found: {path}")

    content = path.read_text(encoding="utf-8")
    return parse_rss_xml(content, default_source=default_source)


def ingest_rss_url(
    url: str,
    timeout: int = 15,
    default_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch a public RSS feed and parse into raw records."""
    xml_text = fetch_rss_feed(url, timeout=timeout)
    return parse_rss_xml(xml_text, default_source=default_source)
