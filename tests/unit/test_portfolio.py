"""Unit tests for portfolio representation and asset validation."""

from __future__ import annotations

from pathlib import Path
import pytest

from finrisk.stress_testing.portfolio import (
    Portfolio,
    PortfolioAsset,
    load_portfolio_csv,
    validate_asset_row,
)


def test_load_sample_portfolio():
    project_root = Path(__file__).resolve().parent.parent.parent
    sample_csv = project_root / "data" / "sample" / "sample_portfolio.csv"
    assert sample_csv.exists()

    portfolio = load_portfolio_csv(sample_csv)
    assert portfolio.asset_count == 10
    assert portfolio.total_market_value == 17000000.0

    counts = portfolio.asset_type_counts
    assert counts["equity"] == 3
    assert counts["corporate_bond"] == 2
    assert counts["government_bond"] == 2
    assert counts["loan"] == 2
    assert counts["derivative"] == 1


def test_validate_derivative_asset():
    """Verify validation of derivative interest rate swap holding."""
    row = {
        "asset_id": "SW_TEST",
        "asset_name": "Test Pay-Fixed Swap",
        "asset_type": "derivative",
        "market_value": "1000000.0",
        "duration": "4.2",
        "notional": "10000000.0",
        "pay_receive": "pay_fixed",
        "dv01": "4200.0",
    }
    asset = validate_asset_row(row)
    assert asset.asset_type == "derivative"
    assert asset.market_value == 1000000.0
    assert asset.notional == 10000000.0
    assert asset.pay_receive == "pay_fixed"
    assert asset.dv01 == 4200.0

    # Invalid direction raises
    with pytest.raises(ValueError, match="Invalid 'pay_receive'"):
        validate_asset_row({
            "asset_id": "SW_BAD",
            "asset_name": "Bad Swap",
            "asset_type": "derivative",
            "market_value": "1000000.0",
            "pay_receive": "straddle",
        })

    # Negative notional raises
    with pytest.raises(ValueError, match="must be strictly positive"):
        validate_asset_row({
            "asset_id": "SW_BAD",
            "asset_name": "Bad Swap",
            "asset_type": "derivative",
            "market_value": "1000000.0",
            "notional": "-500000",
        })


def test_validate_asset_valid():
    row = {
        "asset_id": "EQ_999",
        "asset_name": "Test Equity Corp",
        "asset_type": "equity",
        "market_value": "500000.0",
        "sector": "Technology",
    }
    asset = validate_asset_row(row)
    assert asset.asset_id == "EQ_999"
    assert asset.asset_type == "equity"
    assert asset.market_value == 500000.0
    assert asset.duration is None


def test_validate_asset_missing_required_fields():
    # Missing asset_id
    with pytest.raises(ValueError, match="Missing required 'asset_id'"):
        validate_asset_row({"asset_name": "A", "asset_type": "equity", "market_value": "100"})

    # Missing market_value
    with pytest.raises(ValueError, match="Missing required 'market_value'"):
        validate_asset_row({"asset_id": "1", "asset_name": "A", "asset_type": "equity"})


def test_validate_asset_negative_market_value():
    with pytest.raises(ValueError, match="must be strictly positive"):
        validate_asset_row({"asset_id": "1", "asset_name": "A", "asset_type": "equity", "market_value": "-500"})

    with pytest.raises(ValueError, match="must be strictly positive"):
        validate_asset_row({"asset_id": "1", "asset_name": "A", "asset_type": "equity", "market_value": "0"})


def test_validate_asset_invalid_type():
    with pytest.raises(ValueError, match="Invalid 'asset_type'"):
        validate_asset_row({"asset_id": "1", "asset_name": "A", "asset_type": "crypto", "market_value": "1000"})


def test_validate_asset_invalid_numeric_bounds():
    # Negative duration
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_asset_row({
            "asset_id": "1", "asset_name": "A", "asset_type": "government_bond",
            "market_value": "1000", "duration": "-3.5"
        })

    # PD > 1.0
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        validate_asset_row({
            "asset_id": "1", "asset_name": "A", "asset_type": "loan",
            "market_value": "1000", "baseline_pd": "1.5"
        })


def test_load_portfolio_file_not_found(tmp_path):
    missing_file = tmp_path / "nonexistent_portfolio.csv"
    with pytest.raises(FileNotFoundError):
        load_portfolio_csv(missing_file)
