# FinRisk Intelligence — Advanced NLP Improvement Experiments

This directory contains the isolated, reproducible experiments designed to investigate methods for improving financial sentiment classification—specifically focusing on negative regulatory, legal, and credit-risk signals—without sacrificing performance on general financial text.

## Directory Structure

```text
experiments/finrisk_model_improvements/
├── README.md                                    # This directory overview and replication guide
├── dataset_provenance_manifest.json            # Sources, licenses, and splits for all evaluated data
├── stage1_evaluation_audit.json                # Audit verification of previous metrics and mappings
├── stage2_data_readiness_scan.json             # Scan of available public financial NLP datasets
├── advanced_nlp_experiment_results.json        # Head-to-head metrics, bootstrap CIs, and latency benchmarks
├── stage7_compatibility_report.json            # Numerical, contractual, and architectural isolation checks
├── audit_existing_evaluation.py                # Stage 1 audit script (reproduces previous metrics 100%)
├── assess_data_readiness.py                    # Stage 2 data readiness and HF repository inspection
├── run_ensemble_and_routing_experiments.py     # Stages 4 & 5 ensemble tuning and selective routing
└── test_candidate_integration.py               # Stage 7 isolated integration verification script
```

## Reproduction Instructions

All experiments can be reproduced end-to-end using the project's local virtual environment without modifying production code:

```powershell
# 1. Audit existing evaluation validity and reproduce historical metrics
.\.venv\Scripts\python.exe experiments/finrisk_model_improvements/audit_existing_evaluation.py

# 2. Assess external labelled data availability
.\.venv\Scripts\python.exe experiments/finrisk_model_improvements/assess_data_readiness.py

# 3. Execute ensemble tuning, selective routing, and bootstrap CI evaluation
.\.venv\Scripts\python.exe experiments/finrisk_model_improvements/run_ensemble_and_routing_experiments.py

# 4. Verify model compatibility and contract preservation
.\.venv\Scripts\python.exe experiments/finrisk_model_improvements/test_candidate_integration.py
```

## Summary of Evaluated Candidates

1. **Candidate A (Baseline):** Pretrained `ProsusAI/finbert` (Araci, 2019).
2. **Candidate B (Corporate Fine-Tuned):** Checkpoint from `experiments/finbert_finetuning/best_checkpoint/` (Epoch 1 on Financial PhraseBank).
3. **Candidate E (Dev-Tuned Probability Ensemble):** Linear mixture $P = \alpha P_A + (1 - \alpha) P_C$ with $\alpha^* = 0.80$ selected on development sets.
4. **Candidate S (Source-Aware Gated Routing):** Routes regulatory/central bank sources (`sec_edgar`, `federal_reserve`, `ecb`) to Model A, and corporate news to Model C.

## Production Safety Invariants

- **Production code:** `finrisk/nlp/sentiment_analyzer.py`, `finrisk/risk_engine/alerting.py`, `finrisk/stress_testing/`, API, and Streamlit dashboard are completely UNCHANGED.
- **Test suite:** Full test suite passes: 147 passed, 0 failed, 0 skipped.
- **Git status:** Zero files staged or committed.
