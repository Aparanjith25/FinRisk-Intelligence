# FinRisk Intelligence — FinBERT Fine-Tuning Feasibility, Controlled Experiment & Comparative Evaluation Report

**Project:** FinRisk Intelligence  
**Author:** Senior NLP / Financial Risk Engineering Team  
**Event:** S&P Global & CRISIL Campus Hackathon 2026 (Code to Connect — Phase III)  
**Date:** October 18, 2026  
**Artifact Status:** Official Empirical Evaluation & Model Selection Deliverable  
**Decision Outcome:** **`KEEP CURRENT MODEL` (ProsusAI/finbert Pretrained)**

---

## Executive Summary

To determine whether supervised fine-tuning of `ProsusAI/finbert` could improve financial sentiment analysis for automated credit and market risk stress testing, we conducted a controlled, leakage-resistant experiment comparing three distinct models:

1. **Model A:** Production Pretrained `ProsusAI/finbert` (Araci 2019 / Prosus AI)
2. **Model B:** Deterministic Offline Baseline (Lexicon heuristic fallback)
3. **Model C:** Supervised Fine-Tuned FinBERT Candidate (`experiments/finbert_finetuning/best_checkpoint`)

### Key Empirical Findings:
- **On In-Domain Corporate Earnings News (Financial PhraseBank Test, $N=339$):**
  - Fine-tuning produced a minor marginal gain: Accuracy improved from **97.35%** (Model A) to **98.53%** (Model C), and Macro-F1 rose from **0.9628** to **0.9746** (+0.0118).
- **On Central Bank & Regulatory Oversight Text (Regulatory Benchmark, $N=30$):**
  - Fine-tuning caused **catastrophic domain forgetting and severe performance degradation**.
  - Accuracy **collapsed from 76.67% to 60.00%** (-16.67 percentage points).
  - Macro-F1 **dropped from 0.7548 to 0.5556** (-19.92 percentage points).
  - Most critically for financial risk management, **Negative Class Recall collapsed from 70.0% down to 20.0%**. The fine-tuned candidate misclassified 8 out of 10 severe regulatory enforcement actions (SEC fraud complaints, Federal Reserve cease-and-desist orders) as "neutral" boilerplate.
- **Root Cause of the High Dev Macro-F1 (0.9684):**
  - Pretrained `ProsusAI/finbert` had already achieved **0.9680 Macro-F1 and 97.94% accuracy** on the PhraseBank development split *before fine-tuning started*. Financial PhraseBank is the exact corpus upon which `ProsusAI/finbert` was historically trained in published literature. Fine-tuning on it overfits corporate accounting vocabulary at the expense of regulatory and macro-prudential risk sensitivity.

### Recommendation:
**`KEEP CURRENT MODEL`**. The existing production setup—using pretrained `ProsusAI/finbert` with the deterministic rule-based fallback—is demonstrably more robust, generalized, and safer for portfolio stress testing than the fine-tuned candidate.

---

## 1. Hardware, Data & Computational Feasibility

| Parameter | Specification / Observed Metric |
| :--- | :--- |
| **System Architecture** | Windows AMD64, 16.0 GB RAM (4.25 GB free available) |
| **GPU Hardware** | NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM, Driver 537.53, CUDA 12.2) |
| **Runtime Environment** | Python 3.13.5 (Isolated `.venv`), PyTorch 2.14.1+cpu |
| **Device Used** | Multi-threaded CPU execution (preserving virtualenv stability) |
| **Training Duration** | 935.7 seconds (~15.6 minutes per epoch, 100 steps/epoch, batch size 16) |
| **Total Parameters** | 109,484,547 (BERT-base architecture) |
| **Checkpoint Footprint** | 418.35 MB on disk (`model.safetensors`: 437.96 MB) |
| **Inference Latency** | Baseline: ~0.01 ms \| Pretrained FinBERT: 41.33 ms \| Fine-Tuned FinBERT: 42.86 ms |

---

## 2. Dataset Provenance & Leakage-Resistant Split Design

### 2.1 Dataset Provenance
- **Dataset:** Financial PhraseBank (Malo et al., 2014, Aalto University).
- **Subset Used:** `Sentences_AllAgree.txt` (2,264 sentences where 100% of financial annotators agreed on sentiment).
- **Domain:** Corporate earnings announcements, quarterly financial statements, dividend decisions, and industrial contracts from OMX Helsinki listed firms.

### 2.2 Stratified Split Architecture (Seed 42)
To ensure strict leakage prevention, the 2,264 sentences were stratified by label into three disjoint sets:

| Class | Total | Train (70%) | Dev / Validation (15%) | Held-Out Test (15%) |
| :--- | :---: | :---: | :---: | :---: |
| **Neutral** | 1,391 | 974 | 209 | 208 |
| **Positive** | 570 | 399 | 86 | 85 |
| **Negative** | 303 | 212 | 45 | 46 |
| **Total** | **2,264** | **1,585** | **340** | **339** |

### 2.3 Verification of Zero Data Leakage
- **PhraseBank vs Regulatory Benchmark:** An automated cross-match scan of all 2,264 PhraseBank sentences against all headlines, body texts, and combinations of the 30 regulatory benchmark samples confirmed **exactly 0 string or substring overlaps**.
- **Internal Corpus Redundancy:** Analysis of `Sentences_AllAgree.txt` revealed only 5 duplicate boilerplate lines across the entire corpus of 2,264 sentences (0.29%), confirming that split leakage was effectively zero.

---

## 3. Investigation into the High Development Macro-F1 (0.9684)

During training, Epoch 1 achieved a development Macro-F1 of **0.9684** (Accuracy: 98.24%). We conducted an empirical investigation to determine whether this was caused by data leakage, label bugs, or pretraining convergence:

```text
================================================================================
INVESTIGATION: DATA INTEGRITY & DEV MACRO-F1 ROOT CAUSE
================================================================================
[1] Duplicates in Sentences_AllAgree.txt: 5 / 2,264 (0.29%)
[2] Exact Text Overlap across Splits:
    Dev in Train: 1 / 340 (0.29%)
    Test in Train: 1 / 339 (0.29%)
[3] Pretrained ProsusAI/finbert on Dev Split (Zero-Shot / Baseline):
    Pretrained ProsusAI/finbert Dev Accuracy: 0.9794 (97.94%)
    Pretrained ProsusAI/finbert Dev Macro-F1: 0.9680 (96.80%)
```

### Empirical Conclusion on Dev Metric:
Pretrained `ProsusAI/finbert` **already possessed a 0.9680 Macro-F1 on this exact dataset before training commenced**. Supervised fine-tuning for 1 epoch merely shifted it by **+0.0004** (moving from 333/340 to 334/340 correct predictions). The model was already operating at the theoretical Bayes error rate for this corpus.

---

## 4. Head-to-Head Comparative Evaluation Results

### 4.1 Evaluation on Financial PhraseBank Held-Out Test Set ($N=339$)

| Metric | Model B: Offline Baseline | Model A: Pretrained FinBERT | Model C: Fine-Tuned FinBERT |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 68.14% (231/339) | 97.35% (330/339) | **98.53% (334/339)** |
| **Macro F1** | 0.4265 | 0.9628 | **0.9746** |
| **Macro Precision** | 0.4326 | 0.9519 | **0.9792** |
| **Macro Recall** | 0.4444 | 0.9763 | **0.9704** |
| **Positive F1** | 0.4539 | 0.9643 | **0.9708** |
| **Negative F1** | 0.0000 | 0.9388 | **0.9556** |
| **Neutral F1** | 0.8257 | 0.9854 | **0.9976** |
| **Inference Latency** | **0.01 ms** | 41.33 ms | 42.86 ms |

#### Confusion Matrix — Financial PhraseBank Test Split ($N=339$)

```text
Model B (Baseline):
              Pred: Positive   Pred: Negative   Pred: Neutral
True: Positive       32               8               45
True: Negative       16               0               30
True: Neutral         8               1              199

Model A (Pretrained ProsusAI/finbert):
              Pred: Positive   Pred: Negative   Pred: Neutral
True: Positive       81               3                1
True: Negative        0              46                0
True: Neutral         2               3              203

Model C (Fine-Tuned FinBERT Candidate):
              Pred: Positive   Pred: Negative   Pred: Neutral
True: Positive       83               1                1
True: Negative        3              43                0
True: Neutral         0               0              208
```

---

### 4.2 Evaluation on Curated Regulatory Risk Benchmark ($N=30$)
*(Out-of-domain evaluation on real central bank and securities regulator text: Federal Reserve, SEC, ECB)*

| Metric | Model B: Offline Baseline | Model A: Pretrained FinBERT | Model C: Fine-Tuned FinBERT |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 70.00% (21/30) | **76.67% (23/30)** | 60.00% (18/30) |
| **Macro F1** | 0.6849 | **0.7548** | 0.5556 |
| **Macro Precision** | 0.7561 | **0.8772** | 0.8333 |
| **Macro Recall** | 0.6722 | **0.7333** | 0.5667 |
| **Positive F1** | 0.6154 | **0.6667** | 0.6667 |
| **Negative F1** | 0.7059 | **0.8235** | **0.3333** *(Severe Drop)* |
| **Neutral F1** | 0.7333 | **0.7742** | 0.6667 |
| **Dev Subset (N=22) Acc / F1** | 63.64% / 0.6405 | **77.27% / 0.7694** | 54.55% / 0.5455 |
| **Held-Out ECB (N=8) Acc / F1** | **87.50% / 0.6410** | 75.00% / 0.2857 | 75.00% / 0.2857 |

#### Confusion Matrix — Regulatory Risk Benchmark ($N=30$)

```text
Model B (Baseline):
              Pred: Positive   Pred: Negative   Pred: Neutral
True: Positive        4               0                4
True: Negative        1               6                3
True: Neutral         0               1               11

Model A (Pretrained ProsusAI/finbert):
              Pred: Positive   Pred: Negative   Pred: Neutral
True: Positive        4               0                4
True: Negative        0               7                3
True: Neutral         0               0               12

Model C (Fine-Tuned FinBERT Candidate):
              Pred: Positive   Pred: Negative   Pred: Neutral
True: Positive        4               0                4
True: Negative        0               2                8  <-- MISSED 8 OUT OF 10 NEGATIVES!
True: Neutral         0               0               12
```

---

## 5. Downstream Financial Risk Analysis & Impact

In FinRisk Intelligence, sentiment classification is not an academic vanity metric; it directly drives **automated portfolio risk governance**:
1. **Signal Generation:** Text sentiment produces a bounded score $s \in [-1.0, +1.0]$.
2. **Alert Thresholds:** High-impact negative signals ($s < -0.3$, impact $> 7.0$) trigger immediate operator alerts.
3. **Stress Testing Scenarios:** Negative credit and regulatory signals map into `ScenarioMapper` triggers (e.g. `Credit Spread Widening (+150 bps)`, `Counterparty Default (30% haircut)`).

### Why Model C Fails in Production Risk Management:
When tested on SEC fraud complaints and Federal Reserve enforcement actions:
- **Model A (Pretrained FinBERT)** correctly identified:
  - *"Federal Reserve Board issues enforcement action with Ontario Bancorporation..."* $\rightarrow$ **Negative** ($s = -0.23$)
  - *"SEC Seeks Final Judgment Against Former Western Asset Co-CIO in Cherry-Picking Scheme..."* $\rightarrow$ **Negative** ($s = -0.26$)
  - *"SEC Charges Meyer Global Management With Defrauding Retail Investors..."* $\rightarrow$ **Negative** ($s = -0.76$)
- **Model C (Fine-Tuned Candidate)** classified all of the above as **Neutral** with near-zero scores ($s \in [-0.003, -0.09]$).
- **Consequence:** Model C would suppress critical risk warnings, preventing `ScenarioMapper` from recommending defensive stress tests on exposed bank counterparties.

---

## 6. Inference Contract & API Verification

Both Model A and Model C were verified against the production inference contract:
- **Continuous Score Bounds:** Verified strictly inside $[-1.0, 1.0]$.
- **Probability Sum:** $\sum P = 1.0000 \pm 10^{-4}$.
- **Latency Trade-Off:**
  - Model B: 0.01 ms (sub-millisecond CPU heuristic)
  - Model A: 41.33 ms (production-ready for live batch pipelines)
  - Model C: 42.86 ms (identical compute profile, no speed advantage)

---

## 7. Final Recommendation & Decision

### Recommendation: **`KEEP CURRENT MODEL`**

1. **Empirical Justification:** 
   Fine-tuning `ProsusAI/finbert` on corporate earnings news yielded an insignificant gain on corporate text (+1.18%) while causing **catastrophic forgetting on banking regulatory text**, slashing negative class recall from 70% to 20% and reducing overall benchmark F1 from 0.7548 to 0.5556.
2. **Production Safety:**
   The existing production implementation of `FinBERTSentimentAnalyzer` with `allow_offline_fallback=True` provides the optimal balance of deep financial comprehension (Model A) and guaranteed offline resiliency (Model B).
3. **No Production Code Changes:**
   In accordance with the hackathon rules and sound engineering practice, the production sentiment engine remains untouched.

---

## 8. Prioritized ML Improvement Roadmap (Post-Hackathon)

To genuinely improve financial sentiment analysis beyond `ProsusAI/finbert` without domain collapse, the following roadmap is recommended:

| Priority | Initiative | Rationale | Feasibility |
| :---: | :--- | :--- | :---: |
| **P1** | **Multi-Task Domain Adaptation (Replay Buffer)** | Train jointly on both corporate news (PhraseBank) and regulatory releases (SEC/Fed EDGAR) with an experience replay buffer to prevent catastrophic forgetting. | High |
| **P2** | **Parameter-Efficient Fine-Tuning (LoRA / QLoRA)** | Freeze BERT backbone weights and adapt only low-rank attention adapter matrices ($r=8$, $\alpha=16$). Preserves general pretraining while adapting domain heads. | High |
| **P3** | **Ensemble Sentiment Gating** | Combine FinBERT with rule-based regulatory lexicon overrides: if high-confidence legal terms ("cease and desist", "indictment", "fraud") are detected, gate sentiment negative. | Immediate |
| **P4** | **Aspect-Based Sentiment Analysis (ABSA)** | Extract entity-specific sentiment rather than document-level sentiment (e.g. distinguishing bank health from counterparty default). | Medium |

---
*Report generated and validated autonomously against experimental logs in `experiments/finbert_finetuning/evaluation_results.json`.*
