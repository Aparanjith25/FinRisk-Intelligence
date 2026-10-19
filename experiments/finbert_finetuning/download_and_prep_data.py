import os
import zipfile
from huggingface_hub import hf_hub_download

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading FinancialPhraseBank-v1.0.zip...")
zip_path = hf_hub_download(
    repo_id="takala/financial_phrasebank",
    filename="data/FinancialPhraseBank-v1.0.zip",
    repo_type="dataset",
    local_dir=DATA_DIR,
)
print("Downloaded to:", zip_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR)
print("Extracted files to:", DATA_DIR)

for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        print("  -", os.path.relpath(os.path.join(root, f), DATA_DIR))
