import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Loading ProsusAI/finbert...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.train()

inputs = tokenizer(["Financial operating profit rose by 15% year-over-year."] * 16, return_tensors="pt", padding=True, truncation=True)
labels = torch.tensor([0] * 16)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("Timing 1 training step (batch=16, seq_len=~15)...")
t0 = time.perf_counter()
optimizer.zero_grad()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
t1 = time.perf_counter()
print(f"Step took: {t1 - t0:.2f} seconds. Loss: {loss.item():.4f}")
