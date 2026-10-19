import os
import json
from huggingface_hub import hf_hub_download

out_dir = r"experiments\finrisk_model_improvements\data\sec_sample"
os.makedirs(out_dir, exist_ok=True)

try:
    sec_file = hf_hub_download(
        repo_id="deniks315/fintech-sec-sentiment-sample",
        filename="fintech_dataset_sample.jsonl",
        repo_type="dataset",
        local_dir=out_dir,
    )
    with open(sec_file, "r", encoding="utf-8") as f:
        lines = [json.loads(f.readline()) for _ in range(5)]
    print("Downloaded SEC sample file:", sec_file)
    for l in lines:
        print(l)
except Exception as e:
    print("Download error:", e)
