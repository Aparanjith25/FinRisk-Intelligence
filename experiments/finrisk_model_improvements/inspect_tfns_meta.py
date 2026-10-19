import os
from huggingface_hub import hf_hub_download

out_dir = r"experiments\finrisk_model_improvements\data\tfns"
os.makedirs(out_dir, exist_ok=True)

meta_file = hf_hub_download(
    repo_id="zeroshot/twitter-financial-news-sentiment",
    filename="sent_dataset_meta.txt",
    repo_type="dataset",
    local_dir=out_dir,
)

with open(meta_file, "r", encoding="utf-8") as f:
    print(f.read())
