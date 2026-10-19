"""Streamlit interactive dashboard for FinRisk Intelligence.

Visualizes financial NLP risk signals and strategic portfolio stress test simulations.
Supports local execution and graceful API fallback without displaying fabricated results.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Ensure project root is in sys.path so 'finrisk' is discoverable when launched directly via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths to data and configuration
PORTFOLIO_PATH = PROJECT_ROOT / "data" / "sample" / "sample_portfolio.csv"
SCENARIOS_PATH = PROJECT_ROOT / "configs" / "stress_scenarios.yaml"
RISK_SIGNALS_PATH = PROJECT_ROOT / "data" / "processed" / "risk_signals.csv"
PUBLIC_NEWS_PATH = PROJECT_ROOT / "data" / "processed" / "public_news_records.csv"
PUBLIC_RISK_SIGNALS_PATH = PROJECT_ROOT / "data" / "processed" / "public_risk_signals.csv"
PUBLIC_FEEDS_CONFIG_PATH = PROJECT_ROOT / "configs" / "public_feeds.yaml"

# Graceful optional imports for Streamlit and Plotly
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None  # type: ignore

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None  # type: ignore
    go = None  # type: ignore

import httpx

from finrisk.risk_engine.alerting import DEFAULT_HIGH_IMPACT_THRESHOLD, detect_high_impact_alerts
from finrisk.stress_testing.portfolio import Portfolio, load_portfolio_csv
from finrisk.stress_testing.scenario_engine import ScenarioEngine, StressTestResult, load_scenarios_from_yaml


API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 3.0


def format_pnl_currency(loss_amount: float) -> str:
    """Format an absolute loss amount as a signed P&L currency string.

    Convention from ScenarioEngine:
        loss_amount > 0 represents a loss (reduction in portfolio value).
        loss_amount < 0 represents a gain (increase in portfolio value).
        loss_amount == 0 represents unchanged value.

    Returns:
        '-$123,000.00' for loss_amount > 0
        '+$123,000.00' for loss_amount < 0
        '$0.00' for loss_amount == 0
    """
    val = round(float(loss_amount or 0.0), 2)
    if val > 0:
        return f"-${val:,.2f}"
    elif val < 0:
        return f"+${abs(val):,.2f}"
    else:
        return "$0.00"


def format_pnl_percentage(loss_pct: float) -> str:
    """Format a percentage loss as a signed P&L percentage string.

    Convention from ScenarioEngine:
        loss_pct > 0 represents a percentage loss.
        loss_pct < 0 represents a percentage gain.
        loss_pct == 0 represents unchanged percentage.

    Returns:
        '-4.1%' for loss_pct > 0
        '+4.1%' for loss_pct < 0
        '0.0%' for loss_pct == 0
    """
    val = round(float(loss_pct or 0.0), 2)
    if val > 0:
        return f"-{val}%"
    elif val < 0:
        return f"+{abs(val)}%"
    else:
        return "0.0%"


def fetch_api_or_fallback(
    endpoint: str,
    fallback_callable,
    *args,
    method: str = "GET",
    json_data: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> tuple[Any, str]:
    """Attempt to fetch from local FastAPI server; gracefully fall back to local computation.

    Returns:
        (data, source_description)
    """
    api_url = f"{API_BASE_URL}{endpoint}"
    try:
        if method.upper() == "POST":
            resp = httpx.post(api_url, json=json_data, timeout=DEFAULT_TIMEOUT)
        else:
            resp = httpx.get(api_url, timeout=DEFAULT_TIMEOUT)

        if resp.status_code == 200:
            return resp.json(), f"FastAPI Live Endpoint ({API_BASE_URL})"
        else:
            logger.warning("API endpoint %s returned status %d; falling back to local engine", endpoint, resp.status_code)
    except Exception as exc:
        logger.debug("API connection to %s failed: %s; using local engine", api_url, exc)

    # Safe local fallback
    data = fallback_callable(*args, **kwargs)
    return data, "Local Engine Direct Fallback (API offline - 0 live data fabricated)"


def load_local_risk_signals(
    provenance: str = "synthetic",
    limit: Optional[int] = 500,
) -> List[Dict[str, Any]]:
    """Load risk signals directly from processed CSV files based on requested provenance."""
    signals: List[Dict[str, Any]] = []
    prov_clean = (provenance or "synthetic").lower().strip()
    seen_ids: set[str] = set()

    if prov_clean in ("synthetic", "all") and RISK_SIGNALS_PATH.exists():
        with open(RISK_SIGNALS_PATH, mode="r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r_prov = (r.get("provenance") or "synthetic").strip().lower()
                if r_prov == "synthetic":
                    rid = r.get("record_id")
                    if prov_clean == "all" and rid and rid in seen_ids:
                        continue
                    if rid:
                        seen_ids.add(rid)
                    signals.append(r)

    if prov_clean in ("real_public_news", "public_news", "all") and PUBLIC_RISK_SIGNALS_PATH.exists():
        with open(PUBLIC_RISK_SIGNALS_PATH, mode="r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r_prov = (r.get("provenance") or "real_public_news").strip().lower()
                if r_prov == "real_public_news":
                    rid = r.get("record_id")
                    if prov_clean == "all" and rid and rid in seen_ids:
                        continue
                    if rid:
                        seen_ids.add(rid)
                    signals.append(r)

    if limit is not None and len(signals) > limit:
        signals = signals[:limit]

    return signals


def run_local_public_news_fetch() -> Dict[str, Any]:
    """Execute public news fetching and risk scoring locally."""
    from finrisk.ingestion.public_news import fetch_public_news, save_public_news_csv
    from finrisk.nlp.event_classifier import RuleBasedEventClassifier
    from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer
    from finrisk.risk_engine.engine import RiskEngine

    res = fetch_public_news(config_path=PUBLIC_FEEDS_CONFIG_PATH, timeout=10)
    sig_count = 0
    if res.accepted_records:
        save_public_news_csv(res.accepted_records, PUBLIC_NEWS_PATH)
        engine = RiskEngine(
            sentiment_analyzer=FinBERTSentimentAnalyzer(),
            event_classifier=RuleBasedEventClassifier(),
        )
        signals = engine.process_csv(input_csv=PUBLIC_NEWS_PATH, output_csv=PUBLIC_RISK_SIGNALS_PATH)
        sig_count = len(signals)

    return {
        "status": "success",
        "articles_fetched": res.output_count,
        "signals_generated": sig_count,
        "feed_statuses": [s.to_dict() for s in res.feed_statuses],
    }


def load_local_scenarios() -> Dict[str, Dict[str, Any]]:
    """Load stress scenarios directly from YAML configuration file."""
    if not SCENARIOS_PATH.exists():
        return {}
    return load_scenarios_from_yaml(SCENARIOS_PATH)


def load_local_portfolio() -> Dict[str, Any]:
    """Load baseline portfolio holdings directly from CSV file."""
    if not PORTFOLIO_PATH.exists():
        return {}
    return load_portfolio_csv(PORTFOLIO_PATH).to_dict()


def run_local_stress_test(scenario_id: str) -> Dict[str, Any]:
    """Execute stress test directly using ScenarioEngine."""
    portfolio = load_portfolio_csv(PORTFOLIO_PATH)
    engine = ScenarioEngine.from_config_file(SCENARIOS_PATH)
    result = engine.run_stress_test(portfolio=portfolio, scenario_id=scenario_id)
    return result.to_dict()


def load_local_suggested_scenarios(provenance: str = "all") -> Dict[str, Any]:
    """Analyze local risk signals to suggest stress scenarios."""
    from finrisk.stress_testing.scenario_mapper import recommend_scenarios_from_signals

    signals = load_local_risk_signals(provenance=provenance)
    return recommend_scenarios_from_signals(signals).to_dict()



def main():
    """Main Streamlit dashboard application renderer."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not installed in the current environment.")
        print("To run the interactive dashboard, install streamlit: pip install streamlit")
        return

    st.set_page_config(
        page_title="FinRisk Intelligence",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header and Disclaimers
    st.title("🛡️ FinRisk Intelligence")
    st.caption("AI-Powered Financial Risk Engine & Strategic Portfolio Stress Testing")

    st.warning(
        "⚠️ **PROTOTYPE DEMONSTRATION NOTICE**: Synthetic feeds and portfolio holdings are simulated for demonstration. "
        "When 'Real Public Financial News' is active, genuine public regulatory articles (Fed, SEC, ECB) are ingested. "
        "Impact scores and stress tests are uncalibrated decision-support sensitivity approximations, not econometric predictions."
    )

    # Check API status
    _, api_source_note = fetch_api_or_fallback("/health", lambda: {"status": "offline"})
    st.sidebar.markdown(f"**Data Connectivity:** `{api_source_note}`")
    st.sidebar.divider()

    # Sidebar Data Provenance Toggle
    st.sidebar.subheader("Data Feed Stream")
    if "active_feed_mode" not in st.session_state:
        st.session_state["active_feed_mode"] = "Synthetic Data (Default)"

    feed_mode = st.sidebar.radio(
        "Select Active News/Signal Feed:",
        options=["Synthetic Data (Default)", "Real Public Financial News", "Combined (All Sources)"],
        key="active_feed_mode",
    )
    if feed_mode == "Real Public Financial News":
        prov_param = "real_public_news"
    elif feed_mode == "Combined (All Sources)":
        prov_param = "all"
    else:
        prov_param = "synthetic"

    st.sidebar.divider()
    st.sidebar.subheader("High-Impact Alerts")
    alert_threshold = st.sidebar.slider(
        "Alert Impact Threshold (> X):",
        min_value=1,
        max_value=9,
        value=DEFAULT_HIGH_IMPACT_THRESHOLD,
        step=1,
        help="Flags events with impact scores strictly greater than this threshold (default > 7 as per problem specification).",
    )
    st.sidebar.divider()

    # Tabs
    tab_risk, tab_stress, tab_about = st.tabs([
        "📈 NLP Risk Signals",
        "🎯 Portfolio Stress Testing",
        "ℹ️ Methodology & Caveats",
    ])

    # --- TAB 1: NLP RISK SIGNALS ---
    with tab_risk:
        st.subheader("Financial Text Risk Signals")
        st.caption(f"Active Signal Stream: **{feed_mode}**")

        raw_signals, source_info = fetch_api_or_fallback(
            f"/risk-signals?provenance={prov_param}&limit=500",
            load_local_risk_signals,
            provenance=prov_param,
            limit=500,
        )

        loaded_signals = raw_signals.get("signals", []) if isinstance(raw_signals, dict) else raw_signals

        # Strict provenance enforcement: guarantees synthetic rows are NEVER displayed in Real Public Financial News mode
        if prov_param == "real_public_news":
            signals = [s for s in loaded_signals if (s.get("provenance") or "").strip().lower() == "real_public_news"]
        elif prov_param == "synthetic":
            signals = [s for s in loaded_signals if (s.get("provenance") or "synthetic").strip().lower() == "synthetic"]
        elif prov_param == "all":
            seen_ids = set()
            deduped = []
            for s in loaded_signals:
                rid = s.get("record_id")
                if rid and rid in seen_ids:
                    continue
                if rid:
                    seen_ids.add(rid)
                deduped.append(s)
            signals = deduped
        else:
            signals = loaded_signals

        # Provide on-demand public news fetch button when public or combined feeds are viewed
        if prov_param in ("real_public_news", "all"):
            col_fetch_btn, _ = st.columns([1, 3])
            with col_fetch_btn:
                if st.button("🔄 Fetch Latest Public News", type="secondary"):
                    with st.spinner("Fetching feeds from Federal Reserve, SEC & ECB..."):
                        try:
                            fetch_res, _ = fetch_api_or_fallback(
                                "/public-news/fetch",
                                run_local_public_news_fetch,
                                method="POST",
                            )
                            if isinstance(fetch_res, dict) and fetch_res.get("status") == "success":
                                articles_n = fetch_res.get("articles_fetched", 0)
                                signals_n = fetch_res.get("signals_generated", 0)
                                st.success(f"Fetched {articles_n} articles, generated {signals_n} risk signals!")
                                for s_status in fetch_res.get("feed_statuses", []):
                                    if not s_status.get("success"):
                                        st.warning(f"Feed '{s_status.get('name')}' could not be reached: {s_status.get('error_message')}")
                                st.rerun()
                            else:
                                st.error(f"Failed to fetch public news: {fetch_res}")
                        except Exception as exc:
                            st.error(f"Unexpected error executing public news fetch: {exc}")

        if not signals:
            if prov_param == "real_public_news":
                st.info(
                    "ℹ️ **No real public news articles found in local database.** "
                    "Click **'Fetch Latest Public News'** above or run `python scripts/fetch_public_news.py` "
                    "to ingest live articles from official regulatory feeds (Federal Reserve, SEC, ECB). "
                    "Synthetic records are never substituted."
                )
            elif prov_param == "synthetic":
                st.info("No synthetic risk signals found. Run the synthetic ingestion pipeline first.")
            else:
                st.info("No risk signals found.")
        else:
            # High-Impact Event Alerts (Hackathon Stage 2)
            high_impact_alerts = detect_high_impact_alerts(signals, threshold=alert_threshold)
            if high_impact_alerts:
                st.error(
                    f"🚨 **High-Impact Event Alert ({len(high_impact_alerts)} Detected with Impact Score > {alert_threshold})**"
                )
                with st.expander(f"⚠️ View {len(high_impact_alerts)} High-Impact Risk Alert(s)", expanded=True):
                    for alert in high_impact_alerts:
                        with st.container(border=True):
                            c_ev1, c_ev2, c_ev3 = st.columns([2.5, 1, 1.5])
                            with c_ev1:
                                st.markdown(f"**Event Category:** `{alert.event_category}` | **Source:** `{alert.source}`")
                                st.markdown(f"**Timestamp:** `{alert.published_at}`")
                                st.markdown(f"**Summary:** *{alert.summary_text}*")
                            with c_ev2:
                                st.metric("Impact Score", f"{alert.impact_score}/10")
                                st.caption(f"Sentiment: **{alert.sentiment_label}** ({alert.sentiment_score:+.2f})")
                            with c_ev3:
                                if alert.is_stress_test_applicable and alert.recommended_scenario_id:
                                    st.info(f"**Recommended Scenario:**\n`{alert.recommended_scenario_name}`")
                                    st.caption(f"**Rationale:** {alert.mapping_rationale}")
                                    st.caption("🔒 *Requires explicit user confirmation.*")
                                    if st.button("👉 Stage for Stress Test", key=f"stage_alert_{alert.record_id}"):
                                        st.session_state["selected_scenario_key"] = alert.recommended_scenario_id
                                        st.success(f"Staged '{alert.recommended_scenario_name}'. Switch to Tab 2 to confirm & execute.")
                                else:
                                    st.warning(f"**No Preconfigured Macro Scenario**\n{alert.mapping_rationale}")

            # Metrics
            total_signals = len(signals)
            avg_impact = sum(int(s.get("impact_score", 0)) for s in signals) / total_signals if total_signals else 0.0
            neg_count = sum(1 for s in signals if s.get("sentiment_label") == "negative")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Ingested Signals", total_signals)
            col2.metric("Average Impact Score (1-10)", f"{avg_impact:.2f} / 10")
            col3.metric("Negative Sentiment Signals", f"{neg_count} ({round(neg_count/total_signals*100)}%)")

            # Chart distributions
            st.divider()
            col_chart1, col_chart2 = st.columns(2)

            # Sentiment Distribution
            sentiment_counts = {}
            for s in signals:
                lbl = s.get("sentiment_label", "neutral")
                sentiment_counts[lbl] = sentiment_counts.get(lbl, 0) + 1

            if PLOTLY_AVAILABLE:
                fig_sent = px.pie(
                    names=list(sentiment_counts.keys()),
                    values=list(sentiment_counts.values()),
                    title="Sentiment Class Breakdown",
                    color_discrete_sequence=["#e74c3c", "#2ecc71", "#95a5a6"],
                )
                col_chart1.plotly_chart(fig_sent, use_container_width=True)
            else:
                col_chart1.write(sentiment_counts)

            # Event Category Distribution
            event_counts = {}
            for s in signals:
                ev = s.get("event_type", "Other")
                event_counts[ev] = event_counts.get(ev, 0) + 1

            if PLOTLY_AVAILABLE:
                fig_event = px.bar(
                    x=list(event_counts.keys()),
                    y=list(event_counts.values()),
                    title="Event Categories Distribution",
                    labels={"x": "Event Category", "y": "Signal Count"},
                    color_discrete_sequence=["#2980b9"],
                )
                col_chart2.plotly_chart(fig_event, use_container_width=True)
            else:
                col_chart2.write(event_counts)

            # Data Table
            st.subheader("Signal Feed Details")
            table_records = [
                {
                    "ID": s.get("record_id"),
                    "Provenance": "Real News" if s.get("provenance") == "real_public_news" else "Synthetic",
                    "Source": s.get("source"),
                    "Published": s.get("published_at") or "—",
                    "Company": s.get("company") or "—",
                    "Event Category": s.get("event_type"),
                    "Sentiment": f"{s.get('sentiment_label')} ({float(s.get('sentiment_score', 0)):.2f})",
                    "Impact": f"{s.get('impact_score')}/10",
                    "URL": s.get("url") or "—",
                    "Text Excerpt": s.get("text", "")[:100] + "...",
                }
                for s in signals
            ]
            st.dataframe(table_records, use_container_width=True)

    # --- TAB 2: PORTFOLIO STRESS TESTING ---
    with tab_stress:
        st.subheader("Multi-Asset Portfolio Stress Testing")

        # Baseline Portfolio
        portfolio_data, _ = fetch_api_or_fallback(
            "/portfolio",
            load_local_portfolio,
        )

        total_base_val = portfolio_data.get("total_market_value", 0.0)
        asset_counts = portfolio_data.get("asset_type_counts", {})

        st.metric("Baseline Portfolio Market Value", f"${total_base_val:,.2f}")

        # News-Informed Scenario Suggestion (Decision Support)
        suggested_raw, _ = fetch_api_or_fallback(
            f"/scenarios/suggested?provenance={prov_param}",
            load_local_suggested_scenarios,
            provenance=prov_param,
        )

        with st.expander("💡 News-Informed Scenario Suggestion (Decision Support)", expanded=True):
            top_rec = suggested_raw.get("top_recommendation") if isinstance(suggested_raw, dict) else None
            if top_rec:
                col_rec1, col_rec2 = st.columns([3, 1])
                with col_rec1:
                    st.markdown(
                        f"**Top Recommended Scenario:** :blue-background[**{top_rec.get('scenario_name')}**]  \n"
                        f"**Category:** {top_rec.get('category')} | **Composite Score:** `{top_rec.get('composite_score', 0):.2f}`"
                    )
                with col_rec2:
                    if st.button("👉 Select This Scenario", key="apply_suggested_scenario_btn"):
                        st.session_state["selected_scenario_key"] = top_rec.get("scenario_id")
                        st.rerun()

                st.info(f"📋 **Decision-Support Rationale:** {top_rec.get('rationale')}")

                contribs = top_rec.get("contributing_signals_summary", [])
                if contribs:
                    st.caption(f"Supporting News Headlines ({len(contribs)}):")
                    for c_sig in contribs[:4]:
                        st.markdown(f"- **[{c_sig.get('source')}]** {c_sig.get('headline_excerpt')}")
            else:
                st.caption(
                    "ℹ️ No macroeconomic stress scenario indicated by current news. "
                    "Operational, corporate (M&A), and product events are safely filtered out."
                )

            st.caption(
                "⚠️ *Decision Support Notice: Ingested news events inform scenario selection "
                "but do NOT alter valuation equations or trigger automatic executions. "
                "Review the parameters and execute the test below.*"
            )

        # Scenario Selection
        scenarios_raw, _ = fetch_api_or_fallback(
            "/scenarios",
            load_local_scenarios,
        )

        # Normalize scenario dict whether returned from API (wrapped in {"scenarios": ...})
        # or from local fallback (direct dict of scenarios)
        if isinstance(scenarios_raw, dict):
            if "scenarios" in scenarios_raw and isinstance(scenarios_raw["scenarios"], dict):
                scenarios_dict = scenarios_raw["scenarios"]
            else:
                scenarios_dict = scenarios_raw
        else:
            scenarios_dict = {}

        scenario_keys = [k for k in scenarios_dict.keys() if isinstance(scenarios_dict[k], dict)]

        if not scenario_keys:
            st.error("⚠️ No stress scenarios available. Please verify configs/stress_scenarios.yaml or API availability.")
        else:
            default_index = 0
            preset_key = st.session_state.get("selected_scenario_key")
            if preset_key and preset_key in scenario_keys:
                default_index = scenario_keys.index(preset_key)

            selected_key = st.selectbox(
                "Select Macro Scenario Shock:",
                options=scenario_keys,
                index=default_index,
                format_func=lambda k: scenarios_dict.get(k, {}).get("name", k),
            )

            if selected_key and selected_key in scenarios_dict:
                selected_scenario = scenarios_dict[selected_key]
                st.markdown(f"**Description:** {selected_scenario.get('description', '')}")

                # Run stress test
                if st.button("🚀 Execute Strategic Stress Test", type="primary"):
                    test_res, _ = fetch_api_or_fallback(
                        "/stress-test",
                        run_local_stress_test,
                        selected_key,
                        method="POST",
                        json_data={"scenario_id": selected_key},
                    )

                    stressed_val = test_res.get("stressed_portfolio_value", 0.0)
                    abs_loss = test_res.get("total_absolute_loss", 0.0)
                    pct_loss = test_res.get("total_percentage_loss", 0.0)

                    st.divider()
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "Stressed Valuation",
                        f"${stressed_val:,.2f}",
                        delta=format_pnl_currency(abs_loss),
                        delta_color="inverse",
                    )
                    c2.metric(
                        "Simulated Absolute P&L" if abs_loss < 0 else "Simulated Absolute Loss",
                        f"+${abs(abs_loss):,.2f}" if abs_loss < 0 else f"${abs_loss:,.2f}",
                    )
                    c3.metric(
                        "Simulated Percentage P&L" if pct_loss < 0 else "Simulated Percentage Loss",
                        format_pnl_percentage(pct_loss),
                    )

                    # Loss contribution chart
                    asset_res = test_res.get("asset_results", [])
                    if asset_res and PLOTLY_AVAILABLE:
                        st.subheader("Asset-Level Contribution to Portfolio Loss")
                        asset_names = [a.get("asset_name") for a in asset_res]
                        asset_losses = [a.get("absolute_loss") for a in asset_res]
                        asset_contrib = [a.get("loss_contribution_pct") for a in asset_res]

                        fig_contrib = px.bar(
                            x=asset_names,
                            y=asset_contrib,
                            title="Loss Contribution by Asset (%)",
                            labels={"x": "Portfolio Holding", "y": "Contribution to Total Loss (%)"},
                            color=asset_losses,
                            color_continuous_scale="Reds",
                        )
                        fig_contrib.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_contrib, use_container_width=True)

                    # Asset Results Table
                    st.subheader("Holding Level Valuation Impact")
                    holdings_table = [
                        {
                            "Asset": a.get("asset_name"),
                            "Type": a.get("asset_type"),
                            "Base Value": f"${a.get('baseline_value'):,.2f}",
                            "Stressed Value": f"${a.get('stressed_value'):,.2f}",
                            "Loss ($)": format_pnl_currency(a.get("absolute_loss", 0.0)),
                            "Loss (%)": format_pnl_percentage(a.get("percentage_loss", 0.0)),
                            "Loss Contrib": f"{a.get('loss_contribution_pct')}%",
                            "Method": a.get("valuation_method"),
                        }
                        for a in asset_res
                    ]
                    st.dataframe(holdings_table, use_container_width=True)

    # --- TAB 3: METHODOLOGY ---
    with tab_about:
        st.subheader("Methodology, Modeling Approximations & Limitations")
        st.markdown("""
        ### Valuation Stress Formulas
        1. **Equities**: Direct price shock:
           $$V_{\\text{stressed}} = V_{\\text{base}} \\times (1 + \\Delta_{\\text{equity}})$$
        2. **Government Bonds**: Modified duration sensitivity to interest rates:
           $$V_{\\text{stressed}} = V_{\\text{base}} \\times (1 - D_{\\text{mod}} \\times \\Delta y_{\\text{rates}})$$
        3. **Corporate Bonds**: Duration sensitivity to interest rates plus credit spread widening:
           $$V_{\\text{stressed}} = V_{\\text{base}} \\times (1 - D_{\\text{mod}} \\times (\\Delta y_{\\text{rates}} + \\Delta s_{\\text{spread}}))$$
        4. **Loans**: Baseline probability of default and LGD stressed by scenario factor:
           $$V_{\\text{stressed}} = V_{\\text{base}} \\times (1 - (\\text{PD} \\times \\text{LGD} \\times (1 + \\text{Shock})))$$

        ### Public News Feeds & Provenance
        - **Verified Official Sources**: Ingests public-domain regulatory communications from the **U.S. Federal Reserve** (monetary decisions), the **U.S. SEC** (regulatory & enforcement filings), and the **European Central Bank (ECB)**.
        - **Separation of Datasets**: Public articles are stored and processed in isolated datasets (`public_news_records.csv` and `public_risk_signals.csv`) and never mixed silently with synthetic records.
        - **Zero Web Scraping**: All feeds adhere to public syndication standards with respectful polling intervals and standard User-Agent identification.

        ### News-to-Scenario Decision Support
        - **Explainable Scenario Recommendation**: A transparent rule-based mapper (`ScenarioMapper`) connects categorized risk signals and sentiment severity to the most relevant stylized macro shock scenario (`geopolitical_escalation`, `macroeconomic_stagflation`, `credit_crunch`, `mild_recession`).
        - **User-in-the-Loop Safeguard**: News signals do **not** automatically modify valuation shock parameters or silently execute tests. They provide qualitative decision support; the portfolio manager reviews the contributing evidence and explicitly initiates the stress simulation.
        - **Operational Event Exclusion**: Microeconomic events (M&A, product launches, routine corporate news) are safely filtered out to prevent spurious macro shock triggers.

        ### Systemic Constraints & Modeling Limits
        - **Provisional Heuristics**: Event classification is currently driven by deterministic domain regex patterns, not a fine-tuned deep classification model.
        - **Model Inference Attribution**: Sentiment scores without active transformer weights use offline probability mapping with explicit `offline_fallback_baseline` or `offline_injected_mock` labeling with zero false claims of neural FinBERT inference.
        - **News vs Shock Calibration**: News signals provide qualitative event alerts. They do **not** automatically calibrate quantitative basis-point shocks or predict actual future portfolio losses.
        - **Sensitivity vs Prediction**: Results indicate deterministic sensitivities under stylized scenario shocks, not statistical forecasts.
        """)


if __name__ == "__main__":
    main()
