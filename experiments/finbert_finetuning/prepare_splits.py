import json
import os
import random
from collections import defaultdict

data_file = r"experiments\finbert_finetuning\data\FinancialPhraseBank-v1.0\Sentences_AllAgree.txt"
splits_dir = r"experiments\finbert_finetuning\data\splits"
os.makedirs(splits_dir, exist_ok=True)

# Parse data
samples_by_label = defaultdict(list)
with open(data_file, "r", encoding="latin-1") as f:
    for line in f:
        line = line.strip()
        if not line or "@" not in line:
            continue
        text, label = line.rsplit("@", 1)
        text = text.strip()
        label = label.strip().lower()
        samples_by_label[label].append({"text": text, "label": label})

# Set seed
random.seed(42)

train_samples = []
dev_samples = []
test_samples = []

print("Stratified Splitting (70% Train, 15% Dev, 15% Test) with Seed 42:")
for label, items in sorted(samples_by_label.items()):
    random.shuffle(items)
    n = len(items)
    n_train = int(round(n * 0.70))
    n_dev = int(round(n * 0.15))
    n_test = n - n_train - n_dev

    train_items = items[:n_train]
    dev_items = items[n_train:n_train + n_dev]
    test_items = items[n_train + n_dev:]

    train_samples.extend(train_items)
    dev_samples.extend(dev_items)
    test_samples.extend(test_items)

    print(f"  Class '{label}' (total {n}): train={len(train_items)}, dev={len(dev_items)}, test={len(test_items)}")

# Shuffle final splits
random.shuffle(train_samples)
random.shuffle(dev_samples)
random.shuffle(test_samples)

print(f"Total split sizes: Train={len(train_samples)}, Dev={len(dev_samples)}, Test={len(test_samples)}, Total={len(train_samples)+len(dev_samples)+len(test_samples)}")

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl(os.path.join(splits_dir, "train.jsonl"), train_samples)
save_jsonl(os.path.join(splits_dir, "dev.jsonl"), dev_samples)
save_jsonl(os.path.join(splits_dir, "test.jsonl"), test_samples)
print("Saved splits successfully to:", splits_dir)
