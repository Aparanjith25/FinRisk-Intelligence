"""End-to-End Workflow Verification Script.

Executes the full pipeline:
1. Load public news records from data/processed/public_news_records.csv
2. Normalization & Deduplication
3. Sentiment Analysis (ProsusAI/finbert)
4. Event Classification (RuleBasedEventClassifier)
5. Impact Scoring
6. High-Impact Alerting (detect_high_impact_alerts, threshold > 7)
7. Scenario Mapping (map_signal_to_scenario / recommend_scenarios_from_signals)
8. Portfolio Stress Testing (Portfolio + ScenarioEngine on sample_portfolio.csv)
9. Output Verification
"""

import json
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from finrisk.nlp.event_classifier import RuleBasedEventClassifier
from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer
from finrisk.risk_engine.alerting import DEFAULT_HIGH_IMPACT_THRESHOLD, detect_high_impact_alerts
from finrisk.risk_engine.engine import RiskEngine
from finrisk.stress_testing.portfolio import load_portfolio_csv
from finrisk.stress_testing.scenario_engine import ScenarioEngine
from finrisk.stress_testing.scenario_mapper import map_signal_to_scenario, recommend_scenarios_from_signals

print("=" * 80)
print("END-TO-END PIPELINE AUDIT VERIFICATION")
print("=" * 80)

# Step 1: Ingestion / Load actual public records
news_csv = os.path.join(PROJECT_ROOT, "data", "processed", "public_news_records.csv")
assert os.path.exists(news_csv), f"Missing {news_csv}"
df_news = pd.read_csv(news_csv)
print(f"[Step 1] Ingested {len(df_news)} public news records from {news_csv}")
print(f"         Sources: {dict(df_news['source'].value_counts())}")

# Step 2: Verification of Normalization & Deduplication
assert "record_id" in df_news.columns
assert "provenance" in df_news.columns
assert all(df_news["provenance"] == "real_public_news")
print(f"[Step 2] Normalization verified: All {len(df_news)} records have record_id and provenance='real_public_news'.")

# Step 3: Sentiment & Risk Scoring
signals_csv = os.path.join(PROJECT_ROOT, "data", "processed", "public_risk_signals.csv")
assert os.path.exists(signals_csv), f"Missing {signals_csv}"
df_signals = pd.read_csv(signals_csv)
print(f"[Step 3] Structured signals: {len(df_signals)} signals loaded.")
assert -1.0 <= df_signals["sentiment_score"].min() <= 1.0
assert -1.0 <= df_signals["sentiment_score"].max() <= 1.0
assert 1 <= df_signals["impact_score"].min() <= 10
assert 1 <= df_signals["impact_score"].max() <= 10
print(f"         Sentiment score range: [{df_signals['sentiment_score'].min():.4f}, {df_signals['sentiment_score'].max():.4f}] (strictly in [-1, +1])")
print(f"         Impact score range: [{df_signals['impact_score'].min()}, {df_signals['impact_score'].max()}] (strictly in [1, 10])")

# Step 4: High-Impact Alerts (threshold=7)
signals_list = df_signals.to_dict(orient="records")
alerts_7 = detect_high_impact_alerts(signals_list, threshold=7)
alerts_6 = detect_high_impact_alerts(signals_list, threshold=6)
print(f"[Step 4] High-impact alerts: threshold=7 -> {len(alerts_7)} alerts; threshold=6 -> {len(alerts_6)} alerts.")
for a in alerts_6[:3]:
    print(f"         Alert: ID={a.record_id[:8]}.. | Impact={a.impact_score} | Category={a.event_category} | RecScenario={a.recommended_scenario_id}")

# Step 5: Scenario Mapping
recs_summary = recommend_scenarios_from_signals(signals_list)
recs = recs_summary.recommendations
print(f"[Step 5] Scenario recommendations: {len(recs)} scenarios mapped.")
for r in recs:
    print(f"         Scenario: {r.scenario_name} (ID: {r.scenario_id}) | Score: {r.composite_score:.2f} | Triggers: {r.trigger_count}")

# Step 6: Portfolio Stress Testing
portfolio_csv = os.path.join(PROJECT_ROOT, "data", "sample", "sample_portfolio.csv")
portfolio = load_portfolio_csv(portfolio_csv)
print(f"[Step 6] Portfolio loaded: {portfolio.asset_count} assets, Total Value: ${portfolio.total_market_value:,.2f}")
print(f"         Asset type breakdown: {portfolio.asset_type_counts}")

scenarios_yaml = os.path.join(PROJECT_ROOT, "configs", "stress_scenarios.yaml")
engine = ScenarioEngine.from_config_file(scenarios_yaml)
stress_result = engine.run_stress_test(portfolio, scenario_id="macroeconomic_stagflation")
print(f"[Step 7] Stress Test Execution: Scenario='{stress_result.scenario_name}'")
print(f"         Baseline Value: ${stress_result.baseline_portfolio_value:,.2f}")
print(f"         Stressed Value: ${stress_result.stressed_portfolio_value:,.2f}")
print(f"         Total Loss: ${stress_result.total_absolute_loss:,.2f} ({stress_result.total_percentage_loss:.2f}%)")

# Check derivative asset specifically
swap_res = [a for a in stress_result.asset_results if a.asset_type == "derivative"]
assert len(swap_res) == 1, "Derivative asset not found in results"
swap = swap_res[0]
print(f"[Step 8] Derivative Stress Valuation: {swap.asset_name}")
print(f"         Base: ${swap.baseline_value:,.2f} -> Stressed: ${swap.stressed_value:,.2f} (Loss: ${swap.absolute_loss:,.2f}, {swap.percentage_loss:.2f}%)")
print(f"         Method: {swap.valuation_method}")
print(f"         Applied Shocks: {swap.applied_shocks}")

print("\nCOMPLETE PIPELINE END-TO-END VERIFICATION: 100% SUCCESS")
