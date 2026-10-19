import json
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
try:
    files = api.list_repo_files(repo_id="takala/financial_phrasebank", repo_type="dataset")
    print("Files in repo:", files)
except Exception as e:
    print("Error listing repo:", e)
