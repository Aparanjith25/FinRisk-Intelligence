"""Portfolio representation and validation module.

Loads, validates, and models diversified portfolios across equities, corporate bonds,
government bonds, and loans for strategic stress testing.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

VALID_ASSET_TYPES = ("equity", "corporate_bond", "government_bond", "loan", "derivative")


@dataclass
class PortfolioAsset:
    """Individual portfolio holding with type-specific risk parameters."""

    asset_id: str
    asset_name: str
    asset_type: str  # equity, corporate_bond, government_bond, loan, derivative
    market_value: float  # Current baseline value (must be > 0)
    duration: Optional[float] = None  # Modified/Macaulay duration in years (for bonds/swaps)
    credit_rating: Optional[str] = None  # Credit rating (e.g. AAA, BBB, BB)
    spread_bps: Optional[float] = 0.0  # Credit spread in basis points (for corporate bonds)
    baseline_pd: Optional[float] = None  # Baseline 1-year probability of default [0.0, 1.0]
    lgd: Optional[float] = None  # Loss given default [0.0, 1.0]
    notional: Optional[float] = None  # Underlying contract notional (for derivatives)
    pay_receive: Optional[str] = None  # 'pay_fixed' or 'receive_fixed' (for interest rate swaps)
    dv01: Optional[float] = None  # Dollar Value of an 01 ($/basis point rate shift)
    sector: Optional[str] = None
    company: Optional[str] = None
    is_synthetic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert asset to dictionary representation."""
        return asdict(self)


@dataclass
class Portfolio:
    """Collection of validated portfolio assets."""

    assets: List[PortfolioAsset]

    @property
    def total_market_value(self) -> float:
        """Total baseline portfolio market value."""
        return sum(a.market_value for a in self.assets)

    @property
    def asset_count(self) -> int:
        """Total number of assets."""
        return len(self.assets)

    @property
    def asset_type_counts(self) -> Dict[str, int]:
        """Distribution of holdings by asset class."""
        counts: Dict[str, int] = {}
        for a in self.assets:
            counts[a.asset_type] = counts.get(a.asset_type, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Export portfolio summary and constituent assets."""
        return {
            "total_market_value": round(self.total_market_value, 2),
            "asset_count": self.asset_count,
            "asset_type_counts": self.asset_type_counts,
            "assets": [a.to_dict() for a in self.assets],
        }


def parse_float_safely(val: Any) -> Optional[float]:
    """Safely parse float values, returning None if empty or invalid."""
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def validate_asset_row(row: Dict[str, Any], row_idx: int = 1) -> PortfolioAsset:
    """Validate and construct a PortfolioAsset from a dictionary row.

    Raises:
        ValueError: If required fields are missing, invalid, or out of bounds.
    """
    if not isinstance(row, dict):
        raise ValueError(f"Row {row_idx}: Expected dictionary, got {type(row).__name__}")

    asset_id = str(row.get("asset_id", "")).strip()
    if not asset_id:
        raise ValueError(f"Row {row_idx}: Missing required 'asset_id'")

    asset_name = str(row.get("asset_name", "")).strip()
    if not asset_name:
        raise ValueError(f"Row {row_idx}: Missing required 'asset_name'")

    asset_type = str(row.get("asset_type", "")).strip().lower()
    if asset_type not in VALID_ASSET_TYPES:
        raise ValueError(
            f"Row {row_idx}: Invalid 'asset_type' '{asset_type}'. Must be one of {VALID_ASSET_TYPES}"
        )

    raw_val = row.get("market_value")
    if raw_val is None or str(raw_val).strip() == "":
        raise ValueError(f"Row {row_idx}: Missing required 'market_value'")
    try:
        market_value = float(raw_val)
    except (ValueError, TypeError) as err:
        raise ValueError(f"Row {row_idx}: 'market_value' must be a valid number, got '{raw_val}'") from err

    if market_value <= 0:
        raise ValueError(f"Row {row_idx}: 'market_value' must be strictly positive (> 0), got {market_value}")

    # Type-specific risk parameters
    duration = parse_float_safely(row.get("duration"))
    if duration is not None and duration < 0:
        raise ValueError(f"Row {row_idx}: 'duration' cannot be negative, got {duration}")

    spread_bps = parse_float_safely(row.get("spread_bps"))
    if spread_bps is not None and spread_bps < 0:
        raise ValueError(f"Row {row_idx}: 'spread_bps' cannot be negative, got {spread_bps}")

    baseline_pd = parse_float_safely(row.get("baseline_pd"))
    if baseline_pd is not None and not (0.0 <= baseline_pd <= 1.0):
        raise ValueError(f"Row {row_idx}: 'baseline_pd' must be in [0.0, 1.0], got {baseline_pd}")

    lgd = parse_float_safely(row.get("lgd"))
    if lgd is not None and not (0.0 <= lgd <= 1.0):
        raise ValueError(f"Row {row_idx}: 'lgd' must be in [0.0, 1.0], got {lgd}")

    # Bonds require duration for duration-based approximations
    if asset_type in ("corporate_bond", "government_bond") and duration is None:
        logger.warning(
            "Row %d: %s '%s' missing duration; defaulting to 4.0 years for sensitivity calculations.",
            row_idx, asset_type, asset_id
        )
        duration = 4.0

    # Loans require baseline_pd and lgd for default stress
    if asset_type == "loan":
        if baseline_pd is None:
            baseline_pd = 0.03  # 3% baseline default rate default
        if lgd is None:
            lgd = 0.45  # 45% standard bank LGD default

    # Derivatives (Interest Rate Swaps): parse notional, pay_receive, and dv01
    notional = parse_float_safely(row.get("notional"))
    if notional is not None and notional <= 0:
        raise ValueError(f"Row {row_idx}: 'notional' must be strictly positive (> 0), got {notional}")

    raw_pr = str(row.get("pay_receive", "")).strip().lower()
    pay_receive: Optional[str] = raw_pr or None
    if pay_receive is not None and pay_receive not in ("pay_fixed", "receive_fixed", "payer", "receiver"):
        raise ValueError(
            f"Row {row_idx}: Invalid 'pay_receive' '{pay_receive}'. Expected 'pay_fixed' or 'receive_fixed'"
        )
    if pay_receive in ("payer",):
        pay_receive = "pay_fixed"
    elif pay_receive in ("receiver",):
        pay_receive = "receive_fixed"

    dv01 = parse_float_safely(row.get("dv01"))
    if dv01 is not None and dv01 < 0:
        raise ValueError(f"Row {row_idx}: 'dv01' cannot be negative, got {dv01}")

    if asset_type == "derivative":
        if duration is None:
            duration = 4.2  # 5Y swap standard DV01 duration default
        if notional is None:
            notional = market_value * 10.0  # 10x leverage standard for collateralized IRS default
        if pay_receive is None:
            pay_receive = "pay_fixed"  # Default to pay-fixed macro rate hedge for wholesale bank
        if dv01 is None:
            dv01 = round(notional * duration * 0.0001, 2)

    credit_rating = str(row.get("credit_rating", "")).strip() or None
    sector = str(row.get("sector", "")).strip() or None
    company = str(row.get("company", "")).strip() or None

    is_synth_str = str(row.get("is_synthetic", "true")).strip().lower()
    is_synthetic = is_synth_str in ("true", "1", "yes")

    return PortfolioAsset(
        asset_id=asset_id,
        asset_name=asset_name,
        asset_type=asset_type,
        market_value=market_value,
        duration=duration,
        credit_rating=credit_rating,
        spread_bps=spread_bps or 0.0,
        baseline_pd=baseline_pd,
        lgd=lgd,
        notional=notional,
        pay_receive=pay_receive,
        dv01=dv01,
        sector=sector,
        company=company,
        is_synthetic=is_synthetic,
    )


def load_portfolio_csv(file_path: Union[str, Path]) -> Portfolio:
    """Load and validate a portfolio from a CSV file.

    Args:
        file_path: Path to portfolio CSV.

    Returns:
        Validated Portfolio instance.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If file is empty or contains invalid asset rows.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Portfolio file not found: {path}")

    assets: List[PortfolioAsset] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or headerless CSV file: {path}")

        for idx, row in enumerate(reader, start=1):
            if not any(row.values()):
                continue  # Skip blank lines
            asset = validate_asset_row(row, row_idx=idx)
            assets.append(asset)

    if not assets:
        raise ValueError(f"No valid assets loaded from portfolio CSV: {path}")

    logger.info("Successfully loaded portfolio with %d assets (Total Value: $%.2f)", len(assets), sum(a.market_value for a in assets))
    return Portfolio(assets=assets)
