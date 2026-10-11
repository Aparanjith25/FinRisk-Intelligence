"""FastAPI application for FinRisk Intelligence.

Provides RESTful endpoints for health checks, risk signal retrieval, portfolio
holdings inspection, and scenario stress testing.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path so 'finrisk' is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from finrisk.stress_testing.portfolio import Portfolio, load_portfolio_csv
from finrisk.stress_testing.scenario_engine import ScenarioEngine, load_scenarios_from_yaml

logger = logging.getLogger(__name__)

PORTFOLIO_PATH = PROJECT_ROOT / "data" / "sample" / "sample_portfolio.csv"
SCENARIOS_PATH = PROJECT_ROOT / "configs" / "stress_scenarios.yaml"
RISK_SIGNALS_PATH = PROJECT_ROOT / "data" / "processed" / "risk_signals.csv"
PUBLIC_NEWS_PATH = PROJECT_ROOT / "data" / "processed" / "public_news_records.csv"
PUBLIC_RISK_SIGNALS_PATH = PROJECT_ROOT / "data" / "processed" / "public_risk_signals.csv"
PUBLIC_FEEDS_CONFIG_PATH = PROJECT_ROOT / "configs" / "public_feeds.yaml"

app = FastAPI(
    title="FinRisk Intelligence API",
    version="0.1.0",
    description="AI-powered financial risk engine with strategic portfolio stress testing.",
)


# --- Request and Response Models ---

class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    version: str = Field(..., json_schema_extra={"example": "0.1.0"})
    service: str = Field(..., json_schema_extra={"example": "FinRisk Intelligence API"})


class StressTestRequest(BaseModel):
    scenario_id: Optional[str] = Field(
        None,
        description="ID of preconfigured scenario from configs/stress_scenarios.yaml",
        json_schema_extra={"example": "geopolitical_escalation"},
    )
    custom_parameters: Optional[Dict[str, float]] = Field(
        None,
        description="Custom shock parameters (e.g. {'equity_shock_pct': -15.0, 'interest_rate_shock_bps': 50.0})",
        json_schema_extra={"example": {"equity_shock_pct": -12.5, "interest_rate_shock_bps": 50.0, "credit_spread_shock_bps": 100.0}},
    )
    custom_name: Optional[str] = Field(
        "Custom Shock Scenario",
        description="Optional title for custom scenario run",
    )


def get_scenario_engine() -> ScenarioEngine:
    """Helper to load and return ScenarioEngine."""
    if not SCENARIOS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scenarios configuration file not found at {SCENARIOS_PATH}",
        )
    return ScenarioEngine.from_config_file(SCENARIOS_PATH)


def get_portfolio() -> Portfolio:
    """Helper to load baseline portfolio."""
    if not PORTFOLIO_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Portfolio dataset not found at {PORTFOLIO_PATH}",
        )
    return load_portfolio_csv(PORTFOLIO_PATH)


# --- Endpoints ---

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """Return system operational status and API version."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        service="FinRisk Intelligence API",
    )


@app.get(
    "/scenarios",
    summary="List Stress Scenarios",
    tags=["Stress Testing"],
)
def list_scenarios() -> Dict[str, Any]:
    """Retrieve all available preconfigured stress test scenarios."""
    try:
        scenarios = load_scenarios_from_yaml(SCENARIOS_PATH)
        return {
            "total_scenarios": len(scenarios),
            "scenarios": scenarios,
        }
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load scenarios: {err}",
        ) from err


@app.get(
    "/scenarios/suggested",
    summary="Suggest Stress Scenarios Based on News Signals",
    tags=["Stress Testing"],
)
def get_suggested_scenarios(
    provenance: Optional[str] = Query("all", description="Signal provenance: 'synthetic', 'real_public_news', or 'all'"),
    limit: int = Query(100, ge=1, le=500, description="Max signals to analyze"),
) -> Dict[str, Any]:
    """Analyze processed risk signals and recommend applicable stress test scenarios."""
    from finrisk.stress_testing.scenario_mapper import recommend_scenarios_from_signals

    prov_str = str(provenance) if isinstance(provenance, str) else "all"
    lim_int = int(limit) if isinstance(limit, int) else 100
    raw_res = list_risk_signals(limit=lim_int, company=None, event_type=None, min_impact=None, provenance=prov_str)
    signals = raw_res.get("signals", [])
    summary = recommend_scenarios_from_signals(signals)
    return summary.to_dict()


@app.post(
    "/scenarios/match-signals",
    summary="Match Provided Risk Signals to Stress Scenarios",
    tags=["Stress Testing"],
)
def match_signals_to_scenarios(
    signals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Map user-provided risk signals to recommended portfolio stress test scenarios."""
    from finrisk.stress_testing.scenario_mapper import recommend_scenarios_from_signals

    summary = recommend_scenarios_from_signals(signals)
    return summary.to_dict()



@app.get(
    "/portfolio",
    summary="Get Portfolio Holdings",
    tags=["Portfolio"],
)
def get_portfolio_holdings() -> Dict[str, Any]:
    """Retrieve current baseline portfolio holdings and asset distribution."""
    try:
        portfolio = get_portfolio()
        return portfolio.to_dict()
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load portfolio: {err}",
        ) from err


@app.get(
    "/risk-signals",
    summary="List Processed Risk Signals",
    tags=["Risk Signals"],
)
def list_risk_signals(
    limit: int = Query(50, ge=1, le=500, description="Max records to return"),
    company: Optional[str] = Query(None, description="Filter by company name"),
    event_type: Optional[str] = Query(None, description="Filter by event category"),
    min_impact: Optional[int] = Query(None, ge=1, le=10, description="Minimum impact score (1-10)"),
    provenance: Optional[str] = Query("synthetic", description="Filter by provenance: 'synthetic', 'real_public_news', or 'all'"),
) -> Dict[str, Any]:
    """Retrieve processed financial NLP risk signals with optional filtering."""
    # Sanitize Query defaults if called directly in Python
    if not isinstance(company, str):
        company = None
    if not isinstance(event_type, str):
        event_type = None
    if not isinstance(min_impact, int):
        min_impact = None
    if not isinstance(provenance, str):
        provenance = "synthetic"
    if not isinstance(limit, int):
        limit = 50

    target_files: List[Path] = []
    prov_clean = (provenance or "synthetic").lower().strip()
    if prov_clean in ("synthetic", "all"):
        if RISK_SIGNALS_PATH.exists():
            target_files.append(RISK_SIGNALS_PATH)
    if prov_clean in ("real_public_news", "public_news", "all"):
        if PUBLIC_RISK_SIGNALS_PATH.exists():
            target_files.append(PUBLIC_RISK_SIGNALS_PATH)

    if not target_files:
        return {
            "total_count": 0,
            "provenance": prov_clean,
            "signals": [],
            "message": f"No processed risk signals found for provenance '{prov_clean}'.",
        }

    signals: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for fpath in target_files:
        with open(fpath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_prov = (
                    row.get("provenance") or ("real_public_news" if fpath == PUBLIC_RISK_SIGNALS_PATH else "synthetic")
                ).strip().lower()

                # Strict provenance isolation
                if prov_clean == "synthetic" and row_prov != "synthetic":
                    continue
                if prov_clean in ("real_public_news", "public_news") and row_prov != "real_public_news":
                    continue

                rid = row.get("record_id")
                if rid and rid in seen_ids:
                    continue

                # Apply filters
                if company and (not row.get("company") or company.lower() not in row["company"].lower()):
                    continue
                if event_type and (not row.get("event_type") or event_type.lower() != row["event_type"].lower()):
                    continue
                if min_impact is not None:
                    try:
                        if int(row.get("impact_score", 0)) < min_impact:
                            continue
                    except ValueError:
                        pass

                if rid:
                    seen_ids.add(rid)
                signals.append(row)
                if len(signals) >= limit:
                    break
        if len(signals) >= limit:
            break

    return {
        "total_count": len(signals),
        "provenance": prov_clean,
        "signals": signals,
    }


@app.get(
    "/public-news/sources",
    summary="List Configured Public News Sources",
    tags=["Public News"],
)
def list_public_sources() -> Dict[str, Any]:
    """Retrieve all configured public-domain financial news feeds."""
    from finrisk.ingestion.public_news import load_public_feed_configs

    try:
        configs = load_public_feed_configs(PUBLIC_FEEDS_CONFIG_PATH)
        return {
            "total_sources": len(configs),
            "sources": [
                {
                    "feed_id": c.feed_id,
                    "name": c.name,
                    "url": c.url,
                    "description": c.description,
                    "default_category": c.default_category,
                    "provenance": c.provenance,
                    "enabled": c.enabled,
                }
                for c in configs
            ],
        }
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load public feeds configuration: {err}",
        ) from err


@app.post(
    "/public-news/fetch",
    summary="Fetch Latest Public Financial News",
    tags=["Public News"],
)
def trigger_fetch_public_news() -> Dict[str, Any]:
    """Trigger ingestion from configured public feeds and update public risk signals."""
    from finrisk.ingestion.public_news import fetch_public_news, save_public_news_csv
    from finrisk.nlp.event_classifier import RuleBasedEventClassifier
    from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer
    from finrisk.risk_engine.engine import RiskEngine

    try:
        result = fetch_public_news(config_path=PUBLIC_FEEDS_CONFIG_PATH, timeout=10)
        signals_count = 0
        if result.accepted_records:
            save_public_news_csv(result.accepted_records, PUBLIC_NEWS_PATH)
            engine = RiskEngine(
                sentiment_analyzer=FinBERTSentimentAnalyzer(),
                event_classifier=RuleBasedEventClassifier(),
            )
            signals = engine.process_csv(input_csv=PUBLIC_NEWS_PATH, output_csv=PUBLIC_RISK_SIGNALS_PATH)
            signals_count = len(signals)

        return {
            "status": "success",
            "articles_fetched": result.output_count,
            "signals_generated": signals_count,
            "feed_statuses": [s.to_dict() for s in result.feed_statuses],
        }
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch public news: {err}",
        ) from err


@app.post(
    "/stress-test",
    summary="Run Portfolio Stress Test",
    tags=["Stress Testing"],
)
def run_stress_test_endpoint(request: StressTestRequest) -> Dict[str, Any]:
    """Execute strategic portfolio stress test using a preconfigured or custom scenario."""
    if not request.scenario_id and not request.custom_parameters:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'scenario_id' or 'custom_parameters' in request body.",
        )

    portfolio = get_portfolio()
    engine = get_scenario_engine()

    try:
        result = engine.run_stress_test(
            portfolio=portfolio,
            scenario_id=request.scenario_id,
            custom_parameters=request.custom_parameters,
            custom_name=request.custom_name or "Custom Shock Scenario",
        )
        return result.to_dict()
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if request.scenario_id else status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stress test simulation failed: {err}",
        ) from err
