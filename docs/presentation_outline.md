# FinRisk Intelligence — Presentation Slide Outline

**Event:** S&P Global & CRISIL Campus Hackathon  
**Project:** FinRisk Intelligence — AI-Powered Financial Risk Engine & Strategic Portfolio Stress Testing  
**Audience:** Hackathon Jury, Quantitative Risk Analysts, Credit & Valuation Specialists  

---

### Slide 1: Title & Executive Summary
- **Title:** **FinRisk Intelligence**
- **Subtitle:** Bridging Unstructured Market Narratives and Multi-Asset Portfolio Stress Testing
- **Key Message:** An integrated, end-to-end Python risk engine that transforms financial text (central bank feeds, regulatory filings, synthetic market news) into explainable risk signals and runs deterministic valuation shocks across equities, fixed income, and loan portfolios.
- **Presenter Team & Institution:** Student engineering team, IIT (ISM) Dhanbad.

---

### Slide 2: Problem Statement & Industry Relevance
- **The Core Challenge:**
  - Modern risk analysts face an overwhelming deluge of unstructured text (central bank releases, SEC enforcement actions, macro commentary).
  - Traditional VaR and credit models are lagging indicators that fail to capture sudden narrative shifts in real time.
  - Off-the-shelf LLMs frequently hallucinate corporate entities, confuse dates, and produce opaque black-box outputs unsuitable for regulatory audit.
- **Our Solution:**
  - A modular, provenance-preserving pipeline: normalized data ingestion $\rightarrow$ explainable NLP risk scoring $\rightarrow$ multi-asset valuation shocks $\rightarrow$ interactive dashboard & REST API.

---

### Slide 3: End-to-End System Architecture
- **Ingestion Layer:**
  - Standardized normalization schema with sha256 deterministic IDs and zero entity fabrication.
  - Dual data streams: Controlled synthetic benchmarks + verified official public RSS feeds (Federal Reserve, SEC, ECB).
- **AI/NLP Risk Engine:**
  - ProsusAI/FinBERT sentiment architecture mapped to a continuous risk score: $\text{Score} = P(\text{pos}) - P(\text{neg}) \in [-1.0, 1.0]$.
  - 6-category financial event classification (Geopolitical, Macroeconomic, Credit Event, M&A, Product Launch, Other).
  - Bounded, explainable impact scoring ($1$ to $10$) with explicit rationale auditing.
- **Strategic Stress Testing Engine:**
  - Multi-asset valuation sensitivity approximations across Equities, Government Bonds, Corporate Bonds, and Commercial Loans.
- **Interface Layer:**
  - Production-ready FastAPI service (OpenAPI documented) and interactive Streamlit analytics dashboard.

---

### Slide 4: AI/NLP Risk Engine & Explainable Scoring
- **Continuous Sentiment Mapping:**
  - Rather than discrete binary labels, computes $P(\text{positive}) - P(\text{negative})$ to capture nuanced market sentiment.
  - Transparent deployment: Graceful offline baseline fallback with honest metadata attribution (`offline_fallback_baseline` vs. `pretrained_transformers`).
- **Deterministic Event Classifier:**
  - Regex pattern matchers with precedence ordering covering macroeconomic shifts, credit defaults, tariffs, and corporate restructuring.
- **Explainable Impact Scoring (Scale 1–10):**
  - Category base weight + sentiment directional shift + entity linkage + quantitative numerical evidence.
  - Every score includes a plain-text rationale audit string (e.g., *"Base score 4 for Macroeconomic; +3 for negative sentiment (-0.63); +0 no company identified $\rightarrow$ Final: 4/10"*).

---

### Slide 5: Multi-Asset Strategic Portfolio Stress Testing
- **Transparent Valuation Formulas:**
  - **Equities:** $V_{\text{stressed}} = V_{\text{base}} \times (1 + \Delta_{\text{equity}})$
  - **Government Bonds:** Duration sensitivity to sovereign rates: $V_{\text{stressed}} = V_{\text{base}} \times (1 - D_{\text{mod}} \times \Delta y_{\text{rates}})$
  - **Corporate Bonds:** Rates + credit spread widening: $V_{\text{stressed}} = V_{\text{base}} \times (1 - D_{\text{mod}} \times (\Delta y_{\text{rates}} + \Delta s_{\text{spread}}))$
  - **Loans:** Credit impairment scaling: $V_{\text{stressed}} = V_{\text{base}} \times (1 - (\text{PD} \times \text{LGD} \times (1 + \Delta_{\text{loan\_loss}})))$
- **Preconfigured Macro Scenarios:**
  - Geopolitical Escalation & Trade Shock (8.62% portfolio loss).
  - Macroeconomic Stagflation & Rate Spike (10.43% portfolio loss).
  - Severe Credit Crunch & Spread Widening (9.21% portfolio loss).
  - Mild Recessionary Cooling (1.95% portfolio loss).
- **Mathematical Consistency:** Asset-level losses sum exactly to total portfolio loss with 100% loss contribution accounting.

---

### Slide 6: Data Provenance & Real Public News Integration
- **Dual Stream Separation:**
  - **Synthetic Benchmark:** Fixed control dataset for offline reproducibility and reproducible CI/CD pipelines.
  - **Real Public Financial News:** Official regulatory RSS feeds (U.S. Federal Reserve, U.S. SEC, European Central Bank).
  - Isolated file stores (`public_news_records.csv` and `public_risk_signals.csv`); never mixed silently with synthetic records.
- **Strict Provenance Guarantees:**
  - Real news mode displays only verified public records with canonical source URLs and retrieval timestamps.
  - Empty datasets trigger actionable empty-state notices; synthetic records are never substituted as placeholders.

---

### Slide 7: Engineering Rigor, Honest Limitations & Future Roadmap
- **Engineering Excellence:**
  - **106 passing automated tests** across unit, integration, API, and UI AppTest suites.
  - Zero external network dependency required to run the test suite or launch the core dashboard.
  - Defensive fault tolerance: per-feed network isolation, bounded HTTP timeouts, automatic deduplication.
- **Transparent Limitations (Honest AI Principles):**
  - Prototype sensitivity model: shocks are stylized scenarios, not probabilistic market forecasts or calibrated VaR.
  - News signals provide qualitative event intelligence; they do not automatically calibrate basis-point rate shocks.
- **Future Roadmap:**
  - Fine-tuned domain transformer weights on proprietary earnings transcripts.
  - Multi-factor Monte Carlo simulation engine.
  - Live enterprise webhook integration for real-time risk alerts.
