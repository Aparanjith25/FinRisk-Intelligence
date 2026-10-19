"""Stage 2: Assess Labelled Data Readiness.

Investigates:
1. FiQA Task 1 (Financial Opinion Mining and Sentiment Analysis - microblog & headline sentiment).
2. Twitter Financial News Sentiment / Financial News sentiment datasets.
3. Availability of genuinely labelled regulatory/credit enforcement sentiment datasets.
4. Schema compatibility, license terms, class distribution, and domain alignment.
"""

from __future__ import annotations

import json
import os
import sys
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

datasets_to_check = [
    "zeroshot/twitter-financial-news-sentiment",
    "financial_phrasebank",
    "SALT-NLP/FLUE",
    "StephanAkkerman/financial-tweets-sentiment",
    "FinLLMs/FinancialPhraseBank-v1.0",
]

print("=" * 80)
print("STAGE 2: ASSESSING LABELLED DATASET READINESS")
print("=" * 80)

found_datasets = {}

for ds_id in datasets_to_check:
    print(f"\nChecking dataset repository: {ds_id}...")
    try:
        info = api.dataset_info(repo_id=ds_id)
        files = api.list_repo_files(repo_id=ds_id, repo_type="dataset")
        found_datasets[ds_id] = {
            "status": "AVAILABLE",
            "downloads": getattr(info, "downloads", "N/A"),
            "tags": getattr(info, "tags", []),
            "files": files[:10],
        }
        print(f"  -> Found! Files: {files[:5]}")
    except Exception as e:
        found_datasets[ds_id] = {"status": "UNAVAILABLE", "error": str(e)}
        print(f"  -> Not found or restricted: {e}")

# Check FiQA specifically
print("\nSearching for FiQA sentiment dataset repos on HF...")
try:
    fiqa_repos = list(api.list_datasets(search="fiqa", limit=10))
    fiqa_list = [r.id for r in fiqa_repos if "sentiment" in r.id.lower() or "fiqa" in r.id.lower()]
    print("  FiQA candidate repos:", fiqa_list[:5])
except Exception as e:
    print("  FiQA search error:", e)

# Save report
out_path = os.path.join(os.path.dirname(__file__), "stage2_data_readiness_scan.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(found_datasets, f, indent=2)
print(f"\nSaved data scan results to: {out_path}")
