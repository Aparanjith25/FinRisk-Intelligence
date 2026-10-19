# FinRisk Intelligence — Final Technical Acceptance Audit

**Audit Target:** S&P Global & Crisil Campus Hackathon 2026 / DoSelect Phase III  
**Audit Role:** Strict Senior ML Engineer, Financial Risk Model Validator, Hackathon Evaluator  
**Audit Date:** October 16, 2026  
**Project Directory:** `C:\FinRisk-Intelligence`  
**Execution Environment:** Windows 11, Python 3.13.5, 16 GB RAM, RTX 3050 Laptop GPU, 250+ GB Free Disk Space  
**Repository Working State:** Branch `main`, all pre-existing uncommitted changes preserved, 0 git commits/resets executed.

---

## 1. Executive Verdict

### Verdict: **PASS WITH LIMITATIONS**

The FinRisk Intelligence platform demonstrably satisfies the core end-to-end technical requirements of the competition:
1. Ingests and normalizes multi-source unstructured financial text (news and social media).
2. Generates structured risk signals using genuine `ProsusAI/finbert` NLP inference with continuous sentiment scores in $[-1.0, +1.0]$, explainable event classifications (Geopolitical, Macroeconomic, Credit Event, M&A, Product Launch), and bounded impact scores in $[1, 10]$.
3. Exposes structured data via both high-performance REST API endpoints (`/health`, `/portfolio`, `/scenarios`, `/stress-test`, `/risk-signals`) and persistent CSV/JSON stores.
4. Executes downstream Module B event-driven portfolio stress testing on a wholesale banking portfolio of 10 holdings (\$17,000,000.00 base value) spanning Equities, Corporate Bonds, Government Bonds, Corporate Loans, and an Interest Rate Swap derivative.
5. Implements high-impact event alerts (`impact > 7`) that stage recommended stress scenarios while strictly enforcing human-in-the-loop review and execution confirmation.
6. Displays baseline, stressed portfolio values, and granular holding-level loss contributions in both the interactive Streamlit dashboard and API responses.

**Primary Limitations to Note (Non-fatal, documented by design):**
- **Social Media Source:** Social media ingestion is implemented via local structured CSV parsing with synthetic records (`[SYNTHETIC POST]`), not a live external Twitter/Reddit API stream.
- **News Text Span:** News ingestion extracts and analyzes RSS `<title>` and `<description>` elements rather than scraping full HTML article bodies behind paywalls.
- **Financial Derivative Simplification:** The Interest Rate Swap utilizes a linear first-order DV01 sensitivity model without second-order gamma/convexity, cross-currency basis, or counterparty credit valuation adjustments (CVA/DVA).
- **Benchmark Sample Size:** The curated regulatory sentiment benchmark is preliminary ($N=30$), and while FinBERT achieves 76.7% overall accuracy and 0.7548 Macro F1, on the held-out ECB sub-sample ($n=8$), boilerplate regulatory phrasing led to a higher score for the keyword heuristic.

---

## 2. Official Technical Requirements Matrix

| Requirement | Description | Status | Concrete Evidence |
| :--- | :--- | :---: | :--- |
| **A. Two-Source Ingestion** | Ingest unstructured text from $\ge 2$ distinct sources (e.g. news & social). | **PASS** | `finrisk/ingestion/news_ingestion.py` (RSS/XML parser) and `finrisk/ingestion/social_ingestion.py` (CSV parser) ingest both news and social text, normalizing into unified 7-field schema. Live regulatory news ingestion implemented in `finrisk/ingestion/public_news.py` (SEC, ECB, Fed). |
| **B. Structured Risk Signals** | Generate sentiment in $[-1, +1]$, event classification, and impact in $[1, 10]$. | **PASS** | `RiskEngine` runs `ProsusAI/finbert` yielding continuous $P(\text{pos})-P(\text{neg})$ in $[-1.0, +1.0]$, classifies into 5 categories via `RuleBasedEventClassifier`, and scores impact in $[1, 10]$ with explainable rationales. Tested via `tests/unit/test_risk_engine.py`. |
| **C. Output Availability** | Results accessible via API or files. | **PASS** | REST API endpoints `/risk-signals` and `/portfolio` return JSON; CSV files persist in `data/processed/risk_signals.csv` and `data/processed/public_risk_signals.csv`; JSON benchmark report in `data/processed/sentiment_comparison_results.json`. |
| **D. Downstream Stress Testing** | Module B strategic event-driven portfolio stress testing. | **PASS** | `ScenarioEngine` executes multi-asset factor shocks across 4 preconfigured scenarios (`geopolitical_escalation`, `macroeconomic_stagflation`, `credit_crunch`, `mild_recession`) and custom shocks. Verified in `tests/integration/test_stress_test_integration.py`. |
| **E. Synthetic Wholesale Portfolio** | Multi-asset portfolio with loans, bonds, derivatives. | **PASS** | `data/sample/sample_portfolio.csv` contains 10 assets totaling \$17.0M: 3 Equities (\$5.5M), 2 Corporate Bonds (\$3.0M), 2 Government Bonds (\$5.0M), 2 Syndicated Loans (\$2.5M), and 1 Pay-Fixed Interest Rate Swap (\$1.0M). |
| **F. High-Impact Event Alerts** | Event classification & impact recommend/trigger stress test for high-impact events. | **PASS** | `finrisk/risk_engine/alerting.py` flags events with `impact > 7`, maps category to scenario via `ScenarioMapper`, and presents staged alert in Streamlit dashboard and API with `requires_human_confirmation: True`. |
| **G. Loss & Valuation Display** | Display baseline, stressed values, and losses. | **PASS** | Dashboard Tab 2 and API `/stress-test` display Baseline Value, Stressed Value, Total Absolute Loss, Percentage Loss, and Asset Loss Contributions (e.g. \$17.0M baseline $\rightarrow$ \$15.79M stressed, -\$1.21M loss under Geopolitical Escalation). |

---

## 3. Source-by-Source Ingestion Results & Provenance

| Source Name | Source Type | Fetch/Load Outcome | Records Processed | Provenance Label | Data Modality / Notes |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Sample Financial RSS** | `news` | `Loaded from XML fixture` | 4 accepted, 0 rejected | `synthetic` | Local RSS fixture (`sample_news_rss.xml`). Uses `<title>` + `<description>`. |
| **Sample Financial Social** | `social` | `Loaded from CSV fixture` | 4 accepted, 1 rejected, 1 dup removed | `synthetic` | Local CSV fixture (`sample_social_posts.csv`). Explicitly labeled `[SYNTHETIC POST]`. |
| **U.S. SEC Press Releases** | `public_news` | `Live HTTP 200 OK` | 25 articles | `real_public_news` | Ingested live via `https://www.sec.gov/news/pressreleases.rss`. |
| **European Central Bank (ECB)** | `public_news` | `Live HTTP 200 OK` | 15 articles | `real_public_news` | Ingested live via `https://www.ecb.europa.eu/rss/press.html`. |
| **U.S. Federal Reserve** | `public_news` | `Preserved cached (Live HTTP 404)` | 23 articles | `real_public_news` | Endpoint URL returned 404; existing cached records safely preserved without pipeline failure. |

**Important Audit Findings:**
1. **No Fake Live Social Feed:** The project does not scrape live Twitter/X or Reddit. Social posts are loaded from CSV and honestly labeled `source_type: social_synthetic`, `provenance: synthetic`.
2. **Text Granularity:** RSS ingestion extracts headline `<title>` and summary `<description>` fields ($~30-80$ words). It does not scrape full HTML articles.
3. **Fault-Tolerant Provenance Isolation:** If a feed fails or the network is offline, the system never fabricates articles or substitutes synthetic data into the `real_public_news` store.

---

## 4. End-to-End Workflow Execution & Output Trace

A controlled, un-cached high-impact geopolitical event was passed through the entire live pipeline:

### Input Text:
> *"Critical escalation in Middle East conflict disrupts global oil tanker transit; energy ministers announce emergency crude export embargo as supply contract prices surge 40 percent."*

### Step-by-Step Execution Trace:
1. **Normalization & Deduplication:**  
   - Generated Record ID: `c6b04a49fa885ef3` (deterministic SHA-256 hash).
   - Source: `Reuters World Wire` | Published: `2026-10-15T14:30:00+00:00`.
2. **Sentiment Analysis (`ProsusAI/finbert`):**  
   - Predicted Label: `negative`
   - Continuous Sentiment Score: `-0.9131` (strictly in $[-1.0, +1.0]$)
   - Model Confidence: `0.9312` ($93.1\%$)
   - Inference Source: `pretrained_transformers`
3. **Event Classification:**  
   - Category: `Geopolitical`
4. **Explainable Impact Scoring:**  
   - Impact Score: `8 / 10`
   - Rationale: *"Base score 4 for category 'Geopolitical'; +3 for severe negative sentiment (-0.91); +0 (no specific company identified); +1 for quantitative figures/metrics found in text -> Final impact: 8/10"*
5. **High-Impact Alert Triggering (`impact > 7`):**  
   - Triggered: `True` ($8 > 7$)
   - Recommended Scenario: `Geopolitical Escalation & Trade Shock` (`geopolitical_escalation`)
   - Mapping Rationale: *"Geopolitical risk event with adverse sentiment (-0.91) and elevated impact score (8/10) triggers 'Geopolitical Escalation & Trade Shock'."*
   - Requires Human Confirmation: `True`
6. **Portfolio Stress Simulation (`geopolitical_escalation` on \$17M Portfolio):**  
   - Baseline Portfolio Value: **\$17,000,000.00**
   - Stressed Portfolio Value: **\$15,789,431.25**
   - Total Absolute Loss: **\$1,210,568.75** (Percentage Loss: **7.12%**)
   - Holding Count: 10
7. **Unmapped Event Safety Check (Negative Control):**  
   - Text: *"Tech conglomerate acquires boutique graphic design firm for $12M cash."*  
   - Classification: `Merger/Acquisition`  
   - Is Mapped to Macro Scenario: `False` (`scenario_id: None`)  
   - Unmapped Reason: `Corporate Transaction (M&A) (Microeconomic/Operational)`

---

## 5. Independent Financial Calculation Validation

### 5.1 Valuation Formulas
* **Equity:** $V_{\text{stressed}} = V_{\text{base}} \times (1 + \text{shock}_{\%})$
* **Government Bond:** $V_{\text{stressed}} = V_{\text{base}} \times (1 - D \times \Delta y_{\text{rates}})$
* **Corporate Bond:** $V_{\text{stressed}} = V_{\text{base}} \times (1 - D \times (\Delta y_{\text{rates}} + \Delta s_{\text{spread}}))$
* **Corporate Loan:** $V_{\text{stressed}} = V_{\text{base}} \times (1 - \text{PD} \times \text{LGD} \times (1 + \text{shock}_{\%}))$
* **Interest Rate Swap (Pay-Fixed):** $V_{\text{stressed}} = V_{\text{base}} + (\text{DV01}_{\$} \times \Delta y_{\text{rates, bps}})$
* **Interest Rate Swap (Receive-Fixed):** $V_{\text{stressed}} = V_{\text{base}} - (\text{DV01}_{\$} \times \Delta y_{\text{rates, bps}})$

### 5.2 Independent Verification Against Scenario Engine Outputs

#### A. Rate-Up Scenario (`geopolitical_escalation`: Eq -18%, Rate +40 bps, Spread +120 bps, Loan Loss +15%)

| Holding ID | Asset Name | Asset Type | Baseline Value | Hand-Calculated Stressed Value | Engine Output Value | Absolute Loss | Loss % | Hand Match? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `EQ_001` | AlphaCorp Stock | Equity | \$2,500,000.00 | $\$2.5\text{M} \times (1 - 0.18) = \$2,050,000.00$ | \$2,050,000.00 | +\$450,000.00 | +18.00% | **Exact** |
| `CB_001` | GammaBank Notes ($D=4.8$) | Corp Bond | \$2,000,000.00 | $\$2\text{M} \times (1 - 4.8 \times 0.0160) = \$1,846,400.00$ | \$1,846,400.00 | +\$153,600.00 | +7.68% | **Exact** |
| `GB_001` | US Treasury 10Y ($D=8.2$) | Govt Bond | \$3,000,000.00 | $\$3\text{M} \times (1 - 8.2 \times 0.0040) = \$2,901,600.00$ | \$2,901,600.00 | +\$98,400.00 | +3.28% | **Exact** |
| `LN_001` | CRE Syndicated Loan | Loan | \$1,500,000.00 | $\$1.5\text{M} \times (1 - 0.035 \times 0.45 \times 1.15) = \$1,472,831.25$ | \$1,472,831.25 | +\$27,168.75 | +1.81% | **Exact** |
| `SW_001` | Pay-Fixed IRS ($\text{DV01}=\$4,200$) | Derivative | \$1,000,000.00 | $\$1\text{M} + (\$4,200 \times 40) = \$1,168,000.00$ | \$1,168,000.00 | -\$168,000.00 | -16.80% | **Exact (Gain)** |
| **PORTFOLIO** | **Consolidated (10 Assets)** | **Multi-Asset** | **\$17,000,000.00** | $\sum V_{\text{stressed}} = \mathbf{\$15,789,431.25}$ | **\$15,789,431.25** | **+\$1,210,568.75** | **+7.12%** | **Exact** |

#### B. Rate-Down Scenario (`mild_recession`: Rate -50 bps) for Swap Directionality
* Under `mild_recession`, policy rates are cut by 50 bps ($\Delta y = -50$ bps):
  $$\Delta V = +\$4,200 \times (-50) = -\$210,000.00$$
  $$V_{\text{stressed}} = \$1,000,000.00 - \$210,000.00 = \mathbf{\$790,000.00}$$
  $$\text{Absolute Loss} = \$1,000,000.00 - \$790,000.00 = \mathbf{+\$210,000.00} \quad (\text{Loss of } +21.00\%)$$
  *Engine Output:* Baseline: \$1,000,000.00, Stressed: \$790,000.00, Absolute Loss: +\$210,000.00 (+21.00%). **Exact Match.**

### 5.3 Swap Plausibility & Limitation Analysis
- **Theoretical DV01 Alignment:** For a 5-year swap with \$10,000,000 notional and modified duration of 4.2 years:
  $$\text{DV01} \approx \text{Notional} \times D \times 10^{-4} = \$10,000,000 \times 4.2 \times 0.0001 = \$4,200.00/\text{bp}$$
  The contract's stated sensitivity is mathematically consistent with market standards.
- **Model Limitations:**
  1. *Linearity:* Ignores swap gamma / convexity (accurate for shifts $\le 100$ bps; underestimates extreme non-linear tails).
  2. *Single-Factor Curve:* Assumes parallel shifts; does not capture curve steepening or inversion.
  3. *Zero Credit Valuation Adjustment:* Assumes high-grade counterparty clearing without bilateral credit/debit risk adjustments.

---

## 6. Actual API Smoke-Test Results

Live testing against FastAPI (`http://127.0.0.1:8000`) verified the following responses:

| Endpoint | Method | Status | Verified Response Content / Contract |
| :--- | :---: | :---: | :--- |
| `/health` | `GET` | **200 OK** | `{"status": "ok", "version": "0.1.0", "service": "FinRisk Intelligence API"}` |
| `/portfolio` | `GET` | **200 OK** | Total Market Value: `17000000.0`, Asset Count: `10`, Derivative present: `True`. |
| `/scenarios` | `GET` | **200 OK** | Total Scenarios: `4` (`geopolitical_escalation`, `macroeconomic_stagflation`, `credit_crunch`, `mild_recession`). |
| `/stress-test` | `POST` | **200 OK** | Scenario: `geopolitical_escalation`, Baseline: `17000000.0`, Loss: `1210568.75` (7.12%), Assets: `10`. |
| `/stress-test` (Invalid ID) | `POST` | **404 Not Found** | Correctly rejects unknown scenario with `HTTP 404` and lists available scenario IDs. |
| `/stress-test` (Empty Body) | `POST` | **400 Bad Request** | Correctly rejects empty request with `HTTP 400` detail requirement. |
| `/risk-signals` | `GET` | **200 OK** | Returned 10 records for `provenance=real_public_news`. Verified: `sentiment_score` in $[-1.0, 1.0]$, `impact_score` in $[1, 10]$, `inference_source = pretrained_transformers`. |

---

## 7. Actual Test-Suite Counts & Commands

* **Exact Command Executed:**  
  `.\.venv\Scripts\pytest.exe -v`
* **Test Discovery:** 147 tests across 12 test modules.
* **Actual Test Results:**  
  - **Passed:** **147**
  - **Failed:** **0**
  - **Skipped:** **0**
  - **Errors:** **0**
  - **Execution Time:** 52.73 seconds
* **Warning Summary:** 1 deprecation warning from pytest fixture declaration in `test_finbert_live.py` (`PytestRemovedIn10Warning`), which does not affect test correctness.
* **Model/Network Dependence:**  
  - Live FinBERT tests (`tests/unit/test_finbert_live.py`) executed genuine live PyTorch/Transformers inference using cached local weights on CPU.
  - Integration tests use isolated mocked feeds or local fixtures to guarantee offline reproducibility without network flakiness.

---

## 8. Summary of Changes Made During Audit

| File | Change Made | Rationale |
| :--- | :--- | :--- |
| [`finrisk/dashboard/app.py`](file:///C:/FinRisk-Intelligence/finrisk/dashboard/app.py) | Updated prototype notice banner. | The previous warning claimed *"all news articles are synthetic"*. Replaced with accurate text clarifying that real public regulatory news (Fed, SEC, ECB) is ingested when that feed mode is selected, while portfolio assets and stress parameters remain simulated for demonstration. |

---

## 9. Remaining Technical Blockers & Recommendations

### Severity: Low / Non-Blocking (Enhancements for Post-Hackathon Production)
1. **Federal Reserve Feed URL Stability (Low):**  
   The primary Federal Reserve RSS endpoint (`https://www.federalreserve.gov/feeds/press_all.xml`) currently returns HTTP 404 from the server side. The pipeline gracefully falls back to cached records, but updating to alternate Fed sub-feeds (e.g. monetary policy or regulatory releases) will restore real-time polling.
2. **Pytest Fixture Warning (Low):**  
   `test_finbert_live.py` uses an instance-method class-scoped fixture deprecated in future pytest 10. Decorating with `@classmethod` will remove the single warning.
3. **Second-Order Swap Convexity (Informational):**  
   If rates move by $> 300$ bps, adding a second-order $\frac{1}{2} \times \text{Convexity} \times (\Delta y)^2$ term would increase precision.

---

## 10. Distinctions: Verified Facts vs. Assumptions vs. Unverified Claims

| Item | Fact / Assumption / Claim | Validation Details |
| :--- | :---: | :--- |
| **FinBERT NLP Inference** | **Verified Fact** | Verified by inspecting model weights loading, running 147 pytest tests, and confirming `inference_source: pretrained_transformers` in output CSV. |
| **Two-Source Ingestion** | **Verified Fact** | Verified by running pipeline on RSS XML news and Social Media CSV. |
| **Live Social Media Feed** | **Unverified / Debunked Claim** | There is **NO live Twitter/Reddit scraping**. All social media data is ingested from local CSVs containing clearly labeled synthetic posts. |
| **Full Article HTML Scraping** | **Unverified / Debunked Claim** | RSS ingestion parses `<title>` and `<description>` only; full article HTML bodies are **not** scraped. |
| **Portfolio Valuation (\$17M)** | **Verified Fact** | 10 assets across 5 asset types sum exactly to \$17,000,000.00. Hand calculations match engine output to 2 decimal places. |
| **Stress Shocks Represent Predictions** | **Invalid Assumption (Refuted)** | Stress shocks are explicitly labeled and validated as **sensitivity approximations**, not econometric market predictions. |

---

## Plain-Language Conclusion: Does the Project Satisfy Core Technical Requirements?

**YES. The project demonstrably satisfies all core technical requirements of the hackathon today.**

The entire end-to-end pipeline—from multi-source ingestion, normalization, and genuine FinBERT NLP scoring to high-impact alert generation, decision-support scenario mapping, human-confirmed stress execution, and multi-asset portfolio loss attribution—is fully operational, thoroughly tested, and mathematically verified. The documented limitations (synthetic social posts, RSS summary parsing, and simplified linear swap pricing) are standard for a rapid hackathon demonstration and are disclosed transparently with zero data fabrication.
