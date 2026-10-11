# FinRisk Intelligence — Demonstration Walkthrough Scripts

This guide provides two demonstration scripts:
1. **5-Minute Live Demonstration Script** (tailored for hackathon jury pitches).
2. **15-Minute Recorded Video / Technical Walkthrough Script** (deep architectural dive).

---

## Part 1: 5-Minute Live Demonstration Script

### Timing & Agenda Overview
- **0:00 – 0:45 (45s):** Problem statement & platform overview.
- **0:45 – 1:45 (60s):** Streamlit dashboard: NLP Risk Signals & Dual Stream Provenance.
- **1:45 – 3:00 (75s):** Portfolio Stress Testing: Execution, formulas & loss contribution.
- **3:00 – 4:00 (60s):** FastAPI interactive Swagger docs & live endpoint query.
- **4:00 – 5:00 (60s):** Engineering rigor, 129 passing offline tests (4 live ML tests cleanly skipped), honest limitations, and Q&A transition.

---

### Step-by-Step Speaker Notes

#### 0:00 – 0:45 | Introduction & Motivation
> *"Good morning, respected jury members. In quantitative risk management, financial news and regulatory announcements often carry the earliest warning signs of portfolio distress. However, traditional models rely on lagging financial statements, while off-the-shelf LLMs risk hallucinating entities or operating as black boxes.*
>
> *Today, we present **FinRisk Intelligence**—an open, modular Python financial risk platform built for the S&P Global & CRISIL Campus Hackathon. It connects real-time financial narrative ingestion, explainable NLP risk scoring, and strategic multi-asset portfolio stress testing."*

#### 0:45 – 1:45 | Dashboard Tab 1: NLP Risk Signals & Provenance
> *(Screen: Open Streamlit dashboard at http://localhost:8501 on Tab 1: NLP Risk Signals)*
>
> *"Here is the interactive Streamlit dashboard. Notice the top banner clearly disclosing that this is a demonstration prototype with transparent sensitivity modeling.*
>
> *In the sidebar, we provide a 3-way Data Feed Stream selector: **Synthetic Data (Default)**, **Real Public Financial News**, and **Combined**.*
>
> *When we select 'Real Public Financial News', the dashboard displays genuine, publicly accessible releases from the U.S. Federal Reserve, the SEC, and the European Central Bank. Every single signal retains its canonical source URL, publication timestamp, and provenance badge. Crucially, if no public articles exist, the system shows an informative empty-state notice—synthetic demo records are never silently substituted into the real-news stream.*
>
> *Each record is processed through our NLP Risk Engine: we calculate continuous sentiment in [-1, +1], classify the event category into one of six standard buckets, and assign an explainable impact score from 1 to 10 with a complete rationale audit trail."*

#### 1:45 – 3:00 | Dashboard Tab 2: Portfolio Stress Testing
> *(Screen: Switch to Tab 2: Portfolio Stress Testing)*
>
> *"Now let's move to Tab 2: Multi-Asset Strategic Stress Testing. Our baseline test portfolio holds $16,000,000 across four distinct asset classes: Equities, Corporate Bonds, Government Bonds, and Commercial Loans.*
>
> *Let's select the **'Macroeconomic Stagflation & Rate Spike'** scenario—modeling a 15% equity contraction, a 200 bps interest rate hike, and a 150 bps credit spread widening. Let's click 'Execute Strategic Stress Test'.*
>
> *(Click button)*
>
> *Instantly, we see the stressed valuation drops to $14,330,740, representing an absolute simulated loss of $1,669,260, or 10.43%.*
>
> *Notice our loss contribution chart and valuation table: each asset class uses transparent financial formulas—equities undergo price shocks, government bonds reflect modified duration sensitivity, corporate bonds incorporate credit spread widening, and loans account for stressed probability of default and LGD. The asset-level losses sum mathematically to the total portfolio loss, guaranteeing 100% loss attribution consistency."*

#### 3:00 – 4:00 | FastAPI REST Service
> *(Screen: Switch browser tab to http://127.0.0.1:8000/docs)*
>
> *"FinRisk Intelligence is built as a microservice. All capabilities are exposed via a production-ready FastAPI REST service with complete OpenAPI documentation.*
>
> *Here we have endpoints for health monitoring, portfolio holdings, stress test simulation, public news sources, and filtered risk signals.*
>
> *Let's execute a quick query against `GET /risk-signals?provenance=real_public_news`. As you can see, it returns the structured JSON payload containing real regulatory signals, sentiment scores, and attribution URLs."*

#### 4:00 – 5:00 | Engineering Rigor & Conclusion
> *(Screen: Show terminal with 129 passed tests, 4 skipped)*
>
> *"To ensure enterprise-grade reliability, our codebase contains **133 collected tests—129 passed offline and 4 live FinBERT neural tests cleanly skipped in the primary environment**—covering everything from data normalization, event classification, and financial valuation formulas, to scenario mapping, FastAPI endpoints, and UI AppTests.*
>
> *In adherence to honest AI and financial modeling ethics, our platform operates with zero external network dependencies for testing, provides a deterministic offline fallback when deep learning checkpoints are unavailable, and explicitly distinguishes qualitative narrative signals from calibrated market forecasts.*
>
> *Thank you, and we welcome your questions!"*

---

## Part 2: 15-Minute Recorded Video / Technical Walkthrough Script

### 1. Codebase Architecture & Modular Package Layout (0:00 – 3:30)
- Walk through the directory tree in VS Code / IDE:
  - `finrisk/ingestion/`: `normalizer.py`, `news_ingestion.py`, `social_ingestion.py`, `public_news.py`. Explain how `generate_record_id()` creates deterministic SHA-256 hashes to prevent duplication.
  - `finrisk/nlp/`: `sentiment_analyzer.py` and `event_classifier.py`. Explain continuous score mapping: $P(\text{pos}) - P(\text{neg})$ and deterministic rule hierarchies. Explain `deterministic_baseline_sentiment` fallback.
  - `finrisk/risk_engine/`: `engine.py` and `impact_scorer.py`. Review the rationale audit generator.
  - `finrisk/stress_testing/`: `portfolio.py` and `scenario_engine.py`. Review the four asset-class valuation formulas and configuration-driven YAML loading.
  - `finrisk/api/`: `main.py` FastAPI app.
  - `finrisk/dashboard/`: `app.py` Streamlit app with local engine fallback.

### 2. Live Ingestion & Risk Scoring Pipeline (3:30 – 6:30)
- In terminal, execute:
  ```powershell
  python scripts/run_ingestion.py
  python scripts/run_risk_engine.py
  python scripts/fetch_public_news.py
  ```
- Point out the terminal output:
  - Feed fetch status: U.S. Federal Reserve, U.S. SEC, European Central Bank (HTTP 200).
  - 60 real public news articles ingested and saved to `data/processed/public_news_records.csv`.
  - 60 risk signals generated and saved to `data/processed/public_risk_signals.csv`.
  - Average impact score: 2.73/10.

### 3. API Layer Deep Dive (6:30 – 9:30)
- Start Uvicorn:
  ```powershell
  uvicorn finrisk.api.main:app --port 8000
  ```
- Navigate to `http://127.0.0.1:8000/docs`:
  - Demonstrate `GET /health`.
  - Demonstrate `GET /portfolio` (returns $16M portfolio across 9 assets).
  - Demonstrate `GET /scenarios` (returns all 4 scenarios from YAML).
  - Demonstrate `POST /stress-test` with preconfigured `scenario_id: "credit_crunch"`.
  - Demonstrate `POST /stress-test` with `custom_parameters`:
    ```json
    {
      "custom_parameters": {
        "equity_shock_pct": -25.0,
        "interest_rate_shock_bps": 150.0,
        "credit_spread_shock_bps": 200.0,
        "loan_loss_shock_pct": 30.0
      },
      "custom_name": "Hypothetical Stagflation"
    }
    ```

### 4. Interactive Streamlit Dashboard Walkthrough (9:30 – 12:30)
- Launch Streamlit:
  ```powershell
  streamlit run finrisk/dashboard/app.py
  ```
- Demonstrate Data Feed Stream toggle in sidebar:
  - Synthetic Data: shows synthetic news wire and posts.
  - Real Public Financial News: shows Fed, SEC, ECB articles with direct external links.
  - Combined: shows both without duplicates.
- Click "Fetch Latest Public News" button and show success notification.
- Demonstrate Tab 2 (Portfolio Stress Testing):
  - Run all 4 preconfigured scenarios.
  - Examine the asset results table and explain valuation formulas.
- Review Tab 3 (Methodology & Caveats):
  - Highlight LaTeX formulas, data provenance disclosure, and systemic limits.

### 5. Automated Testing & Verification Review (12:30 – 15:00)
- In terminal, execute:
  ```powershell
  pytest -v
  ```
- Showcase **129 passing tests (4 live ML tests cleanly skipped)** across all modules:
  - Unit tests for ingestion, normalization, sentiment, classification, impact scoring, scenario engine.
  - Integration tests for end-to-end pipeline execution.
  - Regression tests for provenance isolation, offline sentiment fallback, and AppTest UI rendering.
- Summarize honest modeling principles and wrap up.
