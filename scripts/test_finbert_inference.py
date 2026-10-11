"""Verification script for genuine FinBERT inference."""
import os
import sys

# Ensure environment variables point to E: drive
os.environ["HF_HOME"] = r"E:\hf_home"
os.environ["HF_HUB_CACHE"] = r"E:\hf_home\hub"
os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_home"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finrisk.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer

def main():
    print("Initializing FinBERTSentimentAnalyzer...")
    analyzer = FinBERTSentimentAnalyzer()
    
    test_text = "Federal Reserve raises benchmark interest rates by 50 bps, signaling persistent inflation concerns."
    print(f"Testing inference on: '{test_text}'")
    
    result = analyzer.analyze(test_text)
    print("\n--- INFERENCE RESULT ---")
    print(f"Model Name:       {result.model_name}")
    print(f"Inference Source: {result.inference_source}")
    print(f"Is Mock:          {result.is_mock}")
    print(f"Label:            {result.label}")
    print(f"Confidence:       {result.confidence:.4f}")
    print(f"Score:            {result.score:.4f}")
    print(f"Probabilities:    {result.probabilities}")
    print("------------------------\n")
    
    assert result.inference_source == "pretrained_transformers", f"Expected pretrained_transformers, got {result.inference_source}"
    assert result.is_mock is False, "Expected is_mock == False"
    print("SUCCESS: Genuine pretrained FinBERT forward-pass verified!")

if __name__ == "__main__":
    main()
