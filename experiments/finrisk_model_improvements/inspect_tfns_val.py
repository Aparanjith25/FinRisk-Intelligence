import os
import pandas as pd
from huggingface_hub import hf_hub_download

out_dir = r"experiments\finrisk_model_improvements\data\tfns"

val_file = hf_hub_download(
    repo_id="zeroshot/twitter-financial-news-sentiment",
    filename="sent_valid.csv",
    repo_type="dataset",
    local_dir=out_dir,
)

df = pd.read_csv(val_file)
print("TFNS validation shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nClass counts in valid:")
print(df["label"].value_counts())
print("\nFirst 5 rows:")
for _, row in df.head(5).iterrows():
    print(f"  Label {row['label']}: {row['text'][:90]}...")
