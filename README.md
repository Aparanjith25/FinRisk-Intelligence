# FinRisk Intelligence

**AI-Powered Financial Risk Engine & Strategic Portfolio Stress Testing**  
*Developed for the S&P Global & CRISIL Campus Hackathon*

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests Passing](https://img.shields.io/badge/tests-106%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary

**FinRisk Intelligence** is an enterprise-grade financial risk platform that connects real-time unstructured market narratives with quantitative multi-asset portfolio stress testing. The platform ingests both controlled synthetic benchmark data and official regulatory financial news feeds (U.S. Federal Reserve, U.S. SEC, European Central Bank), applies explainable NLP risk scoring, and evaluates deterministic valuation shocks across equities, sovereign debt, corporate credit, and commercial loans.

### Core Capabilities
1. **Multi-Stream Data Ingestion & Deduplication:** Parses RSS/Atom XML news and social media CSVs into a standardized schema with deterministic SHA-256 fingerprinting and strict provenance isolation.
2. **AI/NLP Risk Engine:** Financial sentiment analysis mapped to continuous scores in `[-1.0, 1.0]`, rule-based financial event classification across 6 categories, and explainable integer impact scoring (`1` to `10`) with full audit rationales.
3. **Strategic Portfolio Stress Testing:** Multi-asset valuation sensitivity approximations across Equities, Government Bonds, Corporate Bonds, and Loans under preconfigured macroeconomic, geopolitical, and credit shock scenarios.
4. **REST API & Interactive Dashboard:** Production-ready FastAPI microservice with OpenAPI documentation and an interactive Streamlit analytics dashboard supporting dual-stream data exploration and local fallback execution.
5. **Honest AI & Offline Reliability:** 106 automated offline tests with zero external network dependencies, transparent metadata labeling (`offline_fallback_baseline` vs. `pretrained_transformers`), and explicit non-fabrication guarantees.

---

## 🏗️ Architecture & Directory Layout

```text
FinRisk-Intelligence/
├── configs/
│   ├── public_feeds.yaml           # Verified public RSS feeds (Federal Reserve, SEC, ECB)
│   └── stress_scenarios.yaml       # Macroeconomic, geopolitical, and credit shock parameters
├── data/
│   ├── processed/                  # Normalized records and structured risk signals
│   │   ├── ingested_records.csv    # Synthetic ingested records
│   │   ├── public_news_records.csv # Real public regulatory news records
│   │   ├── public_risk_signals.csv # Real public risk signals
│   │   └── risk_signals.csv        # Synthetic risk signals
│   └── sample/                     # Synthetic seed test fixtures (tracked in Git)
│       ├── sample_news_rss.xml
│       ├── sample_portfolio.csv
│       └── sample_social_posts.csv
├── docs/                           # Submission & demonstration materials
│   ├── demo_walkthrough.md         # 5-minute live pitch & 15-minute recorded demo scripts
│   ├── github_readiness_checklist.md # Final GitHub readiness & secrets audit checklist
│   └── presentation_outline.md     # 5–7 slide presentation outline for jury review
├── finrisk/                        # Core Python application package
│   ├── api/                        # FastAPI service layer
│   │   └── main.py
│   ├── dashboard/                  # Streamlit visual dashboard
│   │   └── app.py
│   ├── ingestion/                  # Data normalizer, RSS parser, CSV loader, public news
│   │   ├── news_ingestion.py
│   │   ├── normalizer.py
│   │   ├── public_news.py
│   │   └── social_ingestion.py
│   ├── nlp/                        # Financial sentiment analysis & event classification
│   │   ├── event_classifier.py
│   │   └── sentiment_analyzer.py
│   ├── risk_engine/                # Impact scoring & signal consolidation
│   │   ├── engine.py
│   │   └── impact_scorer.py
│   └── stress_testing/             # Portfolio valuation & macro scenario engine
│       ├── portfolio.py
│       └── scenario_engine.py
├── scripts/
│   ├── fetch_public_news.py        # Public news ingestion & risk scoring runner
│   ├── run_ingestion.py            # Synthetic pipeline ingestion runner
│   └── run_risk_engine.py          # Synthetic risk engine runner
├── tests/
│   ├── conftest.py
│   ├── fixtures/                   # Offline XML test fixtures (Fed, SEC)
│   ├── integration/                # End-to-end integration test suites
│   └── unit/                       # Component-level unit test suites
├── pyproject.toml
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## 🚀 Quickstart & Setup Instructions

### 1. Environment Setup
```powershell
# 1. Clone repository (or navigate to workspace)
cd FinRisk-Intelligence

# 2. Create Python 3.11 virtual environment
py -3.11 -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install core dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite
The entire test suite executes **100% offline** without model downloads or internet connectivity:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
*Expected Result:* `106 passed in ~7.2s`

### 3. Run Pipeline Batch Scripts
```powershell
# 1. Run Synthetic Ingestion Pipeline (Milestone 1)
.\.venv\Scripts\python.exe scripts/run_ingestion.py

# 2. Run Synthetic AI/NLP Risk Engine (Milestone 2)
.\.venv\Scripts\python.exe scripts/run_risk_engine.py

# 3. Ingest Real Public Regulatory News Feeds (Milestone 4)
.\.venv\Scripts\python.exe scripts/fetch_public_news.py
```

### 4. Start FastAPI REST Service
```powershell
.\.venv\Scripts\uvicorn.exe finrisk.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at: **`http://127.0.0.1:8000/docs`**

### 5. Launch Streamlit Visual Analytics Dashboard
```powershell
.\.venv\Scripts\streamlit.exe run finrisk/dashboard/app.py
```
Interactive dashboard will open in your browser at: **`http://localhost:8501`**

---

## 📡 API Endpoints & Sample Responses

### 1. Health Check
* **Endpoint:** `GET /health`
* **Response:**
  ```json
  {
    "status": "ok",
    "version": "0.1.0",
    "service": "FinRisk Intelligence API"
  }
  ```

### 2. Baseline Portfolio Holdings
* **Endpoint:** `GET /portfolio`
* **Response:**
  ```json
  {
    "asset_count": 9,
    "total_market_value": 16000000.0,
    "asset_type_distribution": {
      "equity": 4500000.0,
      "corporate_bond": 4000000.0,
      "government_bond": 3500000.0,
      "loan": 4000000.0
    }
  }
  ```

### 3. List Preconfigured Stress Scenarios
* **Endpoint:** `GET /scenarios`
* **Response:**
  ```json
  {
    "total_scenarios": 4,
    "scenarios": {
      "geopolitical_escalation": {
        "name": "Geopolitical Escalation & Trade Shock",
        "category": "Geopolitical",
        "parameters": {
          "equity_shock_pct": -18.0,
          "interest_rate_shock_bps": 50.0,
          "credit_spread_shock_bps": 120.0,
          "loan_loss_shock_pct": 25.0
        }
      }
    }
  }
  ```

### 4. Execute Scenario Stress Test
* **Endpoint:** `POST /stress-test`
* **Request Body:**
  ```json
  {
    "scenario_id": "macroeconomic_stagflation"
  }
  ```
* **Response:**
  ```json
  {
    "scenario_id": "macroeconomic_stagflation",
    "scenario_name": "Macroeconomic Stagflation & Rate Spike",
    "category": "Macroeconomic",
    "baseline_portfolio_value": 16000000.0,
    "stressed_portfolio_value": 14330740.0,
    "total_absolute_loss": 1669260.0,
    "total_percentage_loss": 10.43,
    "asset_results": [
      {
        "asset_id": "EQ_001",
        "asset_name": "AlphaCorp Common Stock",
        "asset_type": "equity",
        "baseline_value": 2500000.0,
        "stressed_value": 2125000.0,
        "absolute_loss": 375000.0,
        "percentage_loss": 15.0,
        "loss_contribution_pct": 22.47,
        "valuation_method": "Equity Price Shock: V_stressed = V_base * (1 + shock_pct)"
      }
    ],
    "disclaimer": "Simulated stress test results are deterministic valuation approximations..."
  }
  ```

### 5. Filtered Risk Signals (by Provenance)
* **Endpoint:** `GET /risk-signals?provenance=real_public_news&limit=10`
* **Response:**
  ```json
  {
    "total_count": 10,
    "provenance": "real_public_news",
    "signals": [
      {
        "record_id": "ace881c871481cb4",
        "source": "U.S. Federal Reserve Press Releases",
        "published_at": "2026-10-09T14:00:00+00:00",
        "event_type": "Macroeconomic",
        "sentiment_label": "neutral",
        "sentiment_score": "0.0",
        "impact_score": "4",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/other20261009a.htm",
        "provenance": "real_public_news",
        "inference_source": "offline_fallback_baseline"
      }
    ]
  }
  ```

---

## 🧮 Valuation Stress Testing Methodology

Portfolio shocks are computed across four distinct asset classes using mathematically consistent formulas:

| Asset Class | Valuation Stress Formula | Key Sensitivity Parameters |
| :--- | :--- | :--- |
| **Equities** | $V_{\text{stressed}} = V_{\text{base}} \times (1 + \Delta_{\text{equity}})$ | Direct percentage equity return shock |
| **Government Bonds** | $V_{\text{stressed}} = V_{\text{base}} \times (1 - D_{\text{mod}} \times \Delta y_{\text{rates}})$ | Modified duration ($D_{\text{mod}}$), Sovereign yield shock ($\Delta y_{\text{rates}}$) |
| **Corporate Bonds** | $V_{\text{stressed}} = V_{\text{base}} \times (1 - D_{\text{mod}} \times (\Delta y_{\text{rates}} + \Delta s_{\text{spread}}))$ | Modified duration ($D_{\text{mod}}$), Rates shift ($\Delta y$), Credit spread widening ($\Delta s$) |
| **Loans** | $V_{\text{stressed}} = V_{\text{base}} \times (1 - (\text{PD} \times \text{LGD} \times (1 + \Delta_{\text{loan\_loss}})))$ | Baseline default probability ($\text{PD}$), Loss given default ($\text{LGD}$), Loss multiplier |

### Portfolio Loss Consistency
- **Absolute Portfolio Loss:** $\Delta V = \sum_{i} V_{\text{base}, i} - \sum_{i} V_{\text{stressed}, i}$
- **Percentage Portfolio Loss:** $\Delta V_{\%} = \frac{\Delta V}{\sum_{i} V_{\text{base}, i}} \times 100\%$
- **Asset Loss Contribution:** $\text{Contribution}_i = \frac{\Delta V_i}{\Delta V} \times 100\%$
- **Verification Guarantee:** $\sum_{i} \text{Contribution}_i \equiv 100.0\%$, verified across all scenarios.

### Preconfigured Macro Stress Scenarios
- **Geopolitical Escalation & Trade Shock:** Equity -18%, Rates +50 bps, Spreads +120 bps, Loan loss +25% $\rightarrow$ **-8.62% portfolio loss ($1,378,568.75)**.
- **Macroeconomic Stagflation & Rate Spike:** Equity -15%, Rates +200 bps, Spreads +150 bps, Loan loss +20% $\rightarrow$ **-10.43% portfolio loss ($1,669,260.00)**.
- **Severe Credit Crunch & Spread Widening:** Equity -12%, Rates -50 bps, Spreads +300 bps, Loan loss +60% $\rightarrow$ **-9.21% portfolio loss ($1,473,393.75)**.
- **Mild Recessionary Cooling:** Equity -5%, Rates -75 bps, Spreads +50 bps, Loan loss +10% $\rightarrow$ **-1.95% portfolio loss ($312,406.25)**.

---

## 📰 Data Provenance & Real Public News Feeds

FinRisk Intelligence supports two distinct, isolated data streams:
1. **Synthetic Data Baseline:** Hand-crafted, standardized RSS and social media records designed for deterministic integration and unit testing.
2. **Real Public Regulatory News:** Official RSS syndication feeds from official regulatory authorities:
   - **U.S. Federal Reserve Press Releases:** `https://www.federalreserve.gov/feeds/press_all.xml`
   - **U.S. SEC Press Releases:** `https://www.sec.gov/news/pressreleases.rss`
   - **European Central Bank Press Releases:** `https://www.ecb.europa.eu/rss/press.html`

### Provenance Safeguards
- **Separate File Stores:** Real articles reside in `data/processed/public_news_records.csv` and `data/processed/public_risk_signals.csv`; synthetic data resides in `ingested_records.csv` and `risk_signals.csv`.
- **Zero Hallucination:** Companies and dates are never fabricated; if absent from public announcements, `company` remains `None`.
- **Empty State Notice:** In *Real Public Financial News* mode, if no public articles exist, the dashboard displays an explicit empty-state notice. Synthetic records are never substituted into the real-news stream.
- **Canonical Attribution:** Every public risk signal contains its source URL, publication timestamp, and provenance metadata badge.

---

## ⚠️ Modeling Assumptions & Limitations (Honest AI Principles)

1. **Stylized Sensitivity vs. Calibrated Forecasting:** Portfolio stress testing calculates deterministic holding sensitivities under stylized shocks. It does **not** predict actual market losses, macroeconomic default probabilities, or statistical Value at Risk (VaR).
2. **Qualitative Intelligence vs. Quantitative Calibration:** Financial NLP risk signals highlight emerging operational, credit, or geopolitical events. They do **not** automatically calibrate quantitative basis-point rate shocks.
3. **Provisional NLP Heuristics:** Financial event classifications are driven by deterministic regex rules designed for pipeline validation. Sentiment inference in offline mode utilizes a transparent domain baseline (`offline_fallback_baseline`), explicitly labeled to prevent false claims of active deep learning execution.
4. **Zero Web Scraping / Compliance:** The platform strictly accesses public-domain RSS syndication feeds using respectful bounded timeouts (10s) and standard User-Agent identification. No access controls or paywalls are bypassed.

---

## 📁 Submission Materials & Presentation Resources

- **Presentation Slide Outline:** [`docs/presentation_outline.md`](docs/presentation_outline.md) — Suggested 5–7 slide pitch outline tailored for hackathon jury evaluation.
- **Demonstration Walkthrough Scripts:** [`docs/demo_walkthrough.md`](docs/demo_walkthrough.md) — 5-minute live demonstration script and 15-minute recorded technical walkthrough.
- **GitHub Readiness Checklist:** [`docs/github_readiness_checklist.md`](docs/github_readiness_checklist.md) — Pre-submission verification covering data files, dependencies, `.gitignore`, and zero-secrets audit.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
