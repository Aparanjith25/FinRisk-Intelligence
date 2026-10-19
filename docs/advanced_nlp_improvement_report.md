# FinRisk Intelligence — Advanced NLP Improvement Experiments: Applied Research & Technical Validation Report

**Project:** FinRisk Intelligence  
**Focus Area:** Financial NLP, Domain Adaptation, Regulatory Risk Detection, and Architecture Selection  
**Evaluation Target:** S&P Global & CRISIL Campus Hackathon 2026 (Code to Connect — Phase III)  
**Date:** October 18, 2026  
**Artifact Status:** Official Applied NLP Improvement Research Deliverable  
**Final Decision:** **`MORE LABELLED DATA NEEDED`** *(with `Candidate S: Source-Aware Gated Routing` identified as a promising architectural fallback, but keeping production pretrained `ProsusAI/finbert` unchanged until human-annotated regulatory training data is acquired)*

---

## Executive Summary

This investigation was commissioned to find and experimentally validate the most effective method to improve sentiment classification in FinRisk Intelligence—specifically the detection of negative regulatory enforcement actions and credit-risk events—without degrading performance on corporate earnings and market news.

### Key Conclusions:
1. **Existing Metric Reproduction (Stage 1 Audit):**
   All historical metrics from the previous audit reproduced with 100% mathematical precision from saved predictions:
   - Pretrained FinBERT (Model A): **97.35% accuracy, 0.9628 Macro-F1** on Financial PhraseBank Test; **76.67% accuracy, 0.7548 Macro-F1, 70.0% Negative Recall** on Regulatory Benchmark.
   - Fine-Tuned Checkpoint (Model C): **98.53% accuracy, 0.9746 Macro-F1** on PhraseBank Test; **60.00% accuracy, 0.5556 Macro-F1, 20.0% Negative Recall** on Regulatory Benchmark.
2. **Pretraining Provenance Audit:**
   `ProsusAI/finbert` (Araci, 2019) was originally trained directly on Financial PhraseBank. Consequently, evaluations on Financial PhraseBank test splits represent **in-domain evaluation on the original pretraining distribution** (potentially non-independent). The 30-sample Regulatory Benchmark represents the true out-of-domain transfer test.
3. **Data Readiness Limitation (Stage 2):**
   There is currently **no public, human-annotated financial regulatory sentiment dataset** with sufficient scale for supervised domain adaptation. The project’s 30 regulatory benchmark samples must remain strictly held-out (Safety Rule 9). Fine-tuning on corporate news alone produces catastrophic forgetting of legal and regulatory language. A minimum of **500–1,000 human-annotated regulatory/enforcement sentences** is required before supervised fine-tuning can succeed on regulatory text.
4. **Architectural Candidates Benchmarked (Stages 4 & 5):**
   - **Candidate E (Dev-Tuned Probability Ensemble, $\alpha=0.8$):** Combines 80% Model A + 20% Model C. Achieves 96.21% combined accuracy and 0.9516 Macro-F1, but incurs a **116.9% latency penalty** (76.9 ms vs 35.5 ms) and requires **2x memory** (218M parameters) for only a marginal +0.006 F1 gain.
   - **Candidate S (Source-Aware Gated Routing):** Routes regulatory text (`sec_edgar`, `federal_reserve`, `ecb`) to Model A and corporate text to Model C. Achieves the highest combined performance (**96.75% accuracy, 0.9558 Macro-F1, 89.29% Negative Recall** across 369 test samples) without double-inference latency overhead.
5. **Production Safety Invariant:**
   The production model (`ProsusAI/finbert`), API, dashboard, and test suite (147/147 passing) remain **100% unchanged and untouched**.

---

## 1. Experimental Methods Run vs. Omitted

| Experiment / Method | Execution Status | Rationale / Constraint |
| :--- | :---: | :--- |
| **Stage 1: Metric & Provenance Audit** | **COMPLETED** | Verified label mappings, reproduced all metrics from raw logs, audited PhraseBank pretraining overlap, and verified checkpoint correspondence. |
| **Stage 2: Labelled Data Readiness Scan** | **COMPLETED** | Scanned Hugging Face Hub for FiQA, TFNS, and SEC corpora; identified absence of large human-annotated regulatory sentiment datasets. |
| **Candidate A: Pretrained Baseline** | **COMPLETED** | Fully evaluated across all test splits with 1,000 bootstrap CI iterations. |
| **Candidate B: Fine-Tuned Checkpoint** | **COMPLETED** | Evaluated on untouched test sets with 1,000 bootstrap CI iterations. |
| **Candidate E: Probability Ensemble** | **COMPLETED** | Grid-searched $\alpha \in [0.0, 1.0]$ on Dev sets; selected $\alpha^* = 0.8$; evaluated on final test sets; benchmarked latency and memory overhead. |
| **Candidate S: Source-Aware Gated Routing** | **COMPLETED** | Evaluated metadata-driven routing (Corporate $\rightarrow$ Model C, Regulatory $\rightarrow$ Model A) on combined test benchmark ($N=369$). |
| **Experiment C: Mixed Regulatory Fine-Tuning** | **OMITTED** | **Insufficient labelled regulatory data.** Only 30 benchmark examples exist, which must remain held-out (Rule 9). Generating synthetic labels with LLMs is prohibited (Rule 10). |
| **Experiment D: Class-Weighted Fine-Tuning** | **OMITTED** | Class weighting without diverse regulatory samples would only amplify corporate negative patterns ("operating loss") without teaching legal enforcement concepts. |
| **Experiment F: Parameter-Efficient Fine-Tuning (LoRA)** | **OMITTED** | `peft` is not installed in the local `.venv`. In accordance with Safety Rule 4 & 7, new dependencies were not installed into the working project environment. |

---

## 2. Dataset Provenance and Evaluation Validity

### 2.1 Evaluated Datasets

1. **Financial PhraseBank (`Sentences_AllAgree.txt`)**:
   - **Source:** LexisNexis financial articles covering OMX Helsinki listed companies (Malo et al., 2014).
   - **License:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0).
   - **Size & Classes:** 2,264 sentences (Neutral: 1,391 [61.4%], Positive: 570 [25.2%], Negative: 303 [13.4%]).
   - **Splits:** Train = 1,585 (70%), Dev = 340 (15%), Test = 339 (15%). Stratified by label with `seed=42`.
   - **Pretraining Relationship:** `ProsusAI/finbert` (Araci, 2019) was fine-tuned upstream on this corpus. Thus, the 339-sample test split measures in-domain retention on the pretraining distribution, rather than novel domain generalization.
2. **Twitter Financial News Sentiment (TFNS)**:
   - **Source:** `zeroshot/twitter-financial-news-sentiment` (Hugging Face Datasets).
   - **License:** Open Research / Permissive.
   - **Size & Classes:** 11,932 headlines/tweets (Bearish: 1,789 [15.0%], Bullish: 2,398 [20.1%], Neutral: 7,744 [64.9%]).
   - **Domain:** Stock headlines, analyst upgrades/downgrades, and credit rating agency actions (e.g. Moody's).
3. **Curated Regulatory Benchmark (`ANNOTATED_NEWS_SAMPLES`)**:
   - **Source:** Official press releases from the Federal Reserve Board, SEC Litigation Releases, and European Central Bank monetary policy decisions and speeches.
   - **Size & Classes:** 30 samples (Neutral: 12 [40.0%], Negative: 10 [33.3%], Positive: 8 [26.7%]).
   - **Integrity Status:** Completely independent from training and tuning. Verified 0 string or substring overlaps with Financial PhraseBank.

---

## 3. Head-to-Head Comparative Evaluation & Statistical Analysis

Evaluations were performed across three distinct test splits:
- **PhraseBank Test ($N=339$):** In-domain corporate earnings test set.
- **Regulatory Benchmark ($N=30$):** Out-of-domain central bank and securities enforcement benchmark.
- **Combined Test Benchmark ($N=369$):** Unified test benchmark reflecting production multi-source ingestion.

Statistical uncertainty was estimated using **1,000 bootstrap iterations** to generate 95% Confidence Intervals (CI).

### 3.1 Comparative Performance Summary Table

| Evaluation Split | Metric | Candidate A: Pretrained FinBERT (Baseline) | Candidate B: Fine-Tuned Checkpoint | Candidate E: Dev-Tuned Ensemble ($\alpha=0.8$) | Candidate S: Source-Aware Gated Routing |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PhraseBank Test**<br>($N=339$) | **Accuracy** | 97.35% `[0.956, 0.988]` | **98.53%** `[0.974, 0.997]` | 97.94% `[0.965, 0.994]` | **98.53%** `[0.974, 0.997]` |
| | **Macro F1** | 0.9628 `[0.935, 0.985]` | **0.9746** `[0.951, 0.994]` | 0.9696 `[0.944, 0.990]` | **0.9746** `[0.951, 0.994]` |
| | **Neg F1** | 0.9388 | **0.9556** | 0.9474 | **0.9556** |
| | **Neg Recall** | **100.0%** | 93.48% | 97.83% | 93.48% |
| **Regulatory Benchmark**<br>($N=30$) | **Accuracy** | **76.67%** `[0.600, 0.933]` | 60.00% `[0.433, 0.800]` | **76.67%** `[0.600, 0.933]` | **76.67%** `[0.600, 0.933]` |
| | **Macro F1** | **0.7548** `[0.567, 0.910]` | 0.5556 `[0.348, 0.748]` | **0.7548** `[0.567, 0.910]` | **0.7548** `[0.567, 0.910]` |
| | **Neg F1** | **0.8235** | 0.3333 *(Severe Drop)* | **0.8235** | **0.8235** |
| | **Neg Recall** | **70.00%** `[0.400, 1.000]` | 20.00% `[0.000, 0.500]` | **70.00%** `[0.400, 1.000]` | **70.00%** `[0.400, 1.000]` |
| **Combined Test Benchmark**<br>($N=369$) | **Accuracy** | 95.66% | 95.39% | 96.21% | **96.75%** |
| | **Macro F1** | 0.9456 | 0.9348 | 0.9516 | **0.9558** |
| | **Neg Recall** | **94.64%** | 80.36% | **94.64%** | 89.29% |

---

## 4. Confusion Matrices (Regulatory Benchmark, $N=30$)

```text
Candidate A (Pretrained ProsusAI/finbert):
                 Predicted: Positive   Predicted: Negative   Predicted: Neutral
True: Positive            4                     0                     4
True: Negative            0                     7                     3
True: Neutral             0                     0                    12
Negative Class Recall: 7/10 = 70.0% | Negative Precision: 7/7 = 100.0%

Candidate B (Fine-Tuned Checkpoint):
                 Predicted: Positive   Predicted: Negative   Predicted: Neutral
True: Positive            4                     0                     4
True: Negative            0                     2                     8  <-- 80% of negatives missed!
True: Neutral             0                     0                    12
Negative Class Recall: 2/10 = 20.0% | Negative Precision: 2/2 = 100.0%

Candidate E (Probability Ensemble, alpha=0.8):
                 Predicted: Positive   Predicted: Negative   Predicted: Neutral
True: Positive            4                     0                     4
True: Negative            0                     7                     3
True: Neutral             0                     0                    12
Negative Class Recall: 7/10 = 70.0% | Negative Precision: 7/7 = 100.0%

Candidate S (Source-Aware Gated Routing):
                 Predicted: Positive   Predicted: Negative   Predicted: Neutral
True: Positive            4                     0                     4
True: Negative            0                     7                     3
True: Neutral             0                     0                    12
Negative Class Recall: 7/10 = 70.0% | Negative Precision: 7/7 = 100.0%
```

---

## 5. Latency, Memory, and Resource Trade-offs

Benchmarked on identical CPU hardware (AMD64 16GB RAM, sequence length ~15 tokens, 50 iterations):

| Candidate Architecture | Inference Latency (ms/sample) | Latency Overhead | Model Parameters | Memory Working Set (RAM) |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate A (Pretrained FinBERT)** | **35.47 ms** | Baseline (0.0%) | 109,484,547 | ~418 MB |
| **Candidate B (Fine-Tuned Checkpoint)** | 37.10 ms | +4.6% | 109,484,547 | ~418 MB |
| **Candidate E (Ensemble, $\alpha=0.8$)** | **76.91 ms** | **+116.9%** | 218,969,094 | ~836 MB (2x BERT models) |
| **Candidate S (Source-Aware Routing)** | 35.47 ms – 37.10 ms | 0.0% per call | 218,969,094 | ~836 MB (both models resident) |

### Computational Findings:
- **Ensemble Impracticality:** While Candidate E prevents the severe domain collapse of Candidate B, it doubles inference latency (+116.9%) and RAM footprint for a statistically negligible improvement of **+0.0060 Macro-F1** over the baseline on combined data.
- **Routing Efficiency:** Candidate S delivers the highest combined Macro-F1 (0.9558) while keeping per-request latency identical to single-model inference. However, it requires maintaining two model weights in memory.

---

## 6. Regulatory Negative Recall & Downstream Risk Impact

In FinRisk Intelligence, sentiment classification provides qualitative inputs that gate high-impact event alerts and scenario recommendations:
- **Alert Generation:** Signals with impact $> 7.0$ and negative sentiment trigger operator review in the Streamlit dashboard.
- **Scenario Mapping:** Negative sentiment triggers `ScenarioMapper` credit and geopolitical shocks (`credit_crunch`, `geopolitical_escalation`).

### Downstream Failure of Candidate B:
Under Candidate B, 8 out of 10 negative regulatory events are misclassified as Neutral:
- *"SEC Charges Meyer Global Management and Its CEO With Defrauding Retail Investors"* $\rightarrow$ Candidate B predicts **Neutral** ($s = -0.0989$).
- *"SEC Charges Multiple Entities in Fraud Schemes Totaling at Least $15 Million"* $\rightarrow$ Candidate B predicts **Neutral** ($s = -0.0873$).
- *"Federal Reserve Board announces enforcement action against American Express Company"* $\rightarrow$ Candidate B predicts **Neutral** ($s = -0.0428$).

**Risk Implication:** Deploying Candidate B would cause the risk monitoring system to suppress alerts on severe bank regulatory sanctions, leaving portfolio managers unaware of escalating legal and counterparty credit risks.

---

## 7. Data Readiness Assessment: What is Needed for True Domain Adaptation?

To achieve true domain adaptation without catastrophic forgetting, the project requires:
1. **Target Volume:** An annotated training corpus of **500 to 1,000 regulatory sentences** drawn from:
   - SEC Litigation Releases and Administrative Proceedings (fraud, accounting misstatements, disclosure failures).
   - Federal Reserve Board Enforcement Actions (cease-and-desist orders, civil monetary penalties).
   - FDIC, OCC, and CFTC enforcement actions.
   - Central bank monetary policy decisions and speeches (FOMC, ECB, Bank of England).
2. **Balanced Coverage:** Minimum 200 negative regulatory actions, 200 neutral procedural announcements, and 100 positive approvals/deregulatory actions.
3. **Multi-Domain Replay Buffer:** Training must blend 50% regulatory data with 50% corporate financial text to preserve general financial language capabilities.

---

## 8. Final Decision & Recommendation

### Decision: **`MORE LABELLED DATA NEEDED`**

### Recommendation for Hackathon Evaluation:
1. **Keep Production Setup Unchanged:**
   The existing production architecture—using pretrained `ProsusAI/finbert` with the deterministic rule-based fallback (`allow_offline_fallback=True`)—remains the most robust, generalized, and defensible model for the competition.
2. **Promising Architectural Roadmap (Candidate S):**
   If multi-source specialization is pursued in Phase IV, implement **Candidate S (Source-Aware Gated Routing)**:
   - When text originates from `sec_edgar`, `federal_reserve`, or `ecb`, route inference to the generalized pretrained `ProsusAI/finbert`.
   - When text originates from corporate earnings releases or quarterly reports, route to the fine-tuned corporate model.
3. **No Automatic Deployment:**
   In compliance with Safety Rule 12, no experimental model has been deployed to production. The production code, API, dashboard, and 147-test test suite remain pristine.

---
*Autonomous Applied NLP Research & Validation Report authored for FinRisk Intelligence.*
