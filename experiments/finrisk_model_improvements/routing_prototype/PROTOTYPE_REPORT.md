# Source-Aware Sentiment Router Prototype — Technical Validation & Benchmark Report

**Location:** `experiments/finrisk_model_improvements/routing_prototype/`  
**Date:** October 18, 2026  
**Status:** Isolated Technical Evaluation Deliverable (Production Unmodified)  
**Recommendation:** **`MORE EVIDENCE NEEDED`**

---

## 1. Prototype Architecture & Routing Logic

The `SourceAwareSentimentRouter` was implemented in [`router.py`](./router.py) to explore metadata-driven routing:

```text
Incoming Document (text, source)
               │
               ▼
     [ Normalize Source ]
               │
      ┌────────┴────────┐
      ▼                 ▼
[ Regulatory ]    [ Corporate ]         [ Unknown / Ambiguous ]
(Fed, SEC, ECB)   (PhraseBank, Press)   (None, Unlisted, RSS)
      │                 │                        │
      ▼                 ▼                        ▼
 Model A (Pretrained)  Model C (Fine-Tuned)    Model A (Fallback Safe Default)
```

### Invariants Enforced:
1. **Metadata-Only Routing:** Routing decisions are driven strictly by verified source strings (e.g. `sec_press_releases`, `federal_reserve`, `financial_phrasebank`), never by inferred sentiment or predicted event categories.
2. **Safe Fallback:** Any missing, empty, or unlisted source safely defaults to the robust, generalized pretrained `ProsusAI/finbert`.
3. **Contract Adherence:** Outputs conform exactly to `SentimentResult`:
   $$\text{score} = P(\text{positive}) - P(\text{negative}) \in [-1.0, 1.0]$$
   $$\sum P(\text{class}) = 1.0000 \pm 10^{-3}$$

---

## 2. Test Suite & Invariant Verification

Ran dedicated pytest suite [`test_router.py`](./test_router.py):
- **Command:** `pytest experiments/finrisk_model_improvements/routing_prototype/test_router.py -v`
- **Result:** **6 passed in 13.86s** (0 failed, 0 skipped):
  - `test_regulatory_routing_paths`: PASSED (Fed, SEC, ECB route to `regulatory_model`).
  - `test_corporate_routing_paths`: PASSED (PhraseBank, earnings, PR releases route to `corporate_model`).
  - `test_unknown_and_ambiguous_source_fallback`: PASSED (None, empty, unlisted route to `fallback_model`).
  - `test_score_formula_and_bounds`: PASSED ($s = P(pos) - P(neg)$, bounds verified).
  - `test_edge_cases_handling`: PASSED (Empty string, whitespace, 1,500-word inputs handled cleanly).
  - `test_downstream_scenario_mapper_compatibility`: PASSED (Directly compatible with `map_signal_to_scenario`).

---

## 3. Head-to-Head Comparative Benchmark Results

Evaluated across the exact same held-out test examples in [`benchmark_prototype.py`](./benchmark_prototype.py):

| Evaluation Split | Metric | Original Pretrained FinBERT (Baseline) | Source-Aware Router Prototype | Net Change |
| :--- | :--- | :---: | :---: | :---: |
| **PhraseBank Test**<br>(Corporate News, $N=339$) | **Accuracy** | 97.35% (330/339) | **98.53% (334/339)** | +1.18% (+4 samples) |
| | **Macro F1** | 0.9628 | **0.9746** | +0.0118 |
| | **Negative Recall** | **100.0% (46/46)** | 93.48% (43/46) | **-6.52% (Missed 3 negatives)** |
| **Regulatory Benchmark**<br>(Fed, SEC, ECB, $N=30$) | **Accuracy** | **76.67% (23/30)** | **76.67% (23/30)** | 0.00% (Identical) |
| | **Macro F1** | **0.7548** | **0.7548** | 0.00% (Identical) |
| | **Negative Recall** | **70.00% (7/10)** | **70.00% (7/10)** | 0.00% (Identical) |
| **Unknown Source Fallback**<br>(Regulatory via Unknown, $N=30$) | **Accuracy** | **76.67%** | **76.67%** | 0.00% (Fallback preserved) |
| | **Negative Recall** | **70.00%** | **70.00%** | 0.00% (Fallback preserved) |
| **Combined Benchmark**<br>($N=369$) | **Accuracy** | 95.66% (353/369) | **96.75% (357/369)** | +1.09% |
| | **Macro F1** | 0.9456 | **0.9558** | +0.0102 |
| | **Negative Recall** | **94.64% (53/56)** | 89.29% (50/56) | **-5.35% (Missed 3 negatives)** |

---

## 4. Latency and Resource Overhead

| Resource Metric | Original FinBERT | Source-Aware Router | Overhead / Impact |
| :--- | :---: | :---: | :---: |
| **Inference Latency** | 48.72 ms/sample | 44.16 ms/sample | Identical single-model dispatch (within measurement jitter) |
| **Active Models in RAM** | 1 model | 2 models | **2x Memory footprint** |
| **Parameter Count** | 109,484,547 | 218,969,094 | +109.5M parameters |
| **Disk Footprint** | ~418 MB | ~836 MB | +418 MB on disk |

---

## 5. Critical Generalization Discussion

1. **The Combined Metric Illusion:**
   The apparent improvement in combined Accuracy (96.75% vs 95.66%) and Macro-F1 (0.9558 vs 0.9456) is **almost entirely driven by the 339 corporate PhraseBank sentences**, which comprise 91.9% of the combined test set.
2. **PhraseBank Pretraining Overlap:**
   `ProsusAI/finbert` was pre-trained and fine-tuned on Financial PhraseBank during its initial publication (Araci, 2019). High performance on PhraseBank is therefore in-domain retention on the pretraining distribution, not proof of generalized superior reasoning.
3. **Negative Class Degradation on Corporate Text:**
   While Model C gains 4 correct positive/neutral classifications on corporate text, it **misses 3 negative corporate sentences** that the original FinBERT correctly classified, causing corporate negative recall to drop from 100.0% to 93.48%. For a financial risk management system, missing adverse credit or earnings signals is costlier than a minor gain in positive precision.
4. **Small Regulatory Benchmark ($N=30$):**
   The 30-sample regulatory set cannot statistically justify deploying a two-model infrastructure. On regulatory text, the router simply preserves Model A's existing performance.

---

## 6. Final Recommendation

### **`MORE EVIDENCE NEEDED`**

### Detailed Rationale:
- **Pros:** The prototype works cleanly, respects all architectural contracts, enforces safe fallback, and prevents the domain collapse of Model C on regulatory news.
- **Cons:** It requires maintaining **two 110M parameter transformer models in memory (836 MB RAM)**, and on corporate text, Model C actually **reduced negative recall by 6.52%**.
- **Action:** Retain the single pretrained `ProsusAI/finbert` baseline in production until a dedicated regulatory training set (>500 samples) and a larger benchmark (>100 samples) can be collected.
