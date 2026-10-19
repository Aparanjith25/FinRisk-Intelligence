import os
from collections import Counter

data_dir = r"experiments\finbert_finetuning\data\FinancialPhraseBank-v1.0"
files = [
    "Sentences_AllAgree.txt",
    "Sentences_75Agree.txt",
    "Sentences_66Agree.txt",
    "Sentences_50Agree.txt"
]

for fname in files:
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        continue
    with open(fpath, "r", encoding="latin-1") as f:
        lines = f.readlines()
    labels = []
    for l in lines:
        parts = l.strip().rsplit("@", 1)
        if len(parts) == 2:
            labels.append(parts[1].lower())
    c = Counter(labels)
    print(f"File: {fname} | Total: {len(labels)} | Counts: {dict(c)}")
