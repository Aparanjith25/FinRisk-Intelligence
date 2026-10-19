import os
import sys

sys.path.insert(0, os.path.abspath("."))
from tests.evaluation.annotated_news_sample import ANNOTATED_NEWS_SAMPLES

# Load sentences
data_file = r"experiments\finbert_finetuning\data\FinancialPhraseBank-v1.0\Sentences_AllAgree.txt"
with open(data_file, "r", encoding="latin-1") as f:
    pb_sentences = [l.strip().rsplit("@", 1)[0].strip().lower() for l in f if "@" in l]

print(f"Total PhraseBank sentences: {len(pb_sentences)}")

benchmark_texts = []
for s in ANNOTATED_NEWS_SAMPLES:
    benchmark_texts.append(s.headline.strip().lower())
    benchmark_texts.append(s.text.strip().lower())
    benchmark_texts.append((s.headline.strip() + ". " + s.text.strip()).lower())

overlaps = []
for idx, pb_s in enumerate(pb_sentences):
    for b_idx, b_txt in enumerate(benchmark_texts):
        if pb_s in b_txt or b_txt in pb_s:
            overlaps.append((idx, pb_s, b_txt))

print(f"Exact or substring overlaps found: {len(overlaps)}")
if overlaps:
    for o in overlaps[:5]:
        print("Overlap:", o)
else:
    print("Zero overlap confirmed! Datasets are completely disjoint.")
