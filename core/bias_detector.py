import os
from typing import Dict, List

import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class BiasDetector:
    def __init__(self, model_path: str = "."):
        """Load a text classification model and tokenizer for bias detection.

        The detector will try to load from `model_path` first, and fall back to
        a standard DistilBERT checkpoint if loading from the provided path
        fails. The model is moved to GPU if available.

        Args:
            model_path: Local directory or model identifier for from_pretrained.
        """
        print("Loading bias detection model...")

        try:
            # Try to load from specified path first
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            print(f"✅ Loaded model from: {model_path}")
        except Exception as e:
            print(f"❌ Could not load model from {model_path}: {e}")
            print("🔄 Falling back to default model...")
            # Fallback to a base model
            self.model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=2
            )
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            print("✅ Loaded default model")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        print(f"Bias detector loaded on: {self.device}")

    def predict_bias(self, text: str) -> Dict:
        """Return a bias prediction for a single input string.

        The returned dict contains 'bias_probability' (float), 'classification'
        (str) and 'confidence' (float). The classification threshold is 0.5.

        Args:
            text: Input text to classify.

        Returns:
            Dict with keys: 'bias_probability', 'classification', 'confidence'.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            bias_prob = probs[0, 1].item()

        return {
            "bias_probability": bias_prob,
            "classification": "BIASED" if bias_prob > 0.5 else "NEUTRAL",
            "confidence": max(bias_prob, 1 - bias_prob),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Simple per-item prediction for a list of texts.

        This is a convenience wrapper that calls `predict_bias` sequentially.

        Args:
            texts: List of input strings.

        Returns:
            List of prediction dicts as returned by `predict_bias`.
        """
        return [self.predict_bias(text) for text in texts]

    def predict_batch_batched(
        self, texts: List[str], batch_size: int = 8
    ) -> List[Dict]:
        """Predict bias for multiple texts using batched model inference.

        This function tokenizes a batch of texts and runs them through the
        model with a single forward pass per batch for improved throughput.

        Args:
            texts: List of input strings.
            batch_size: Number of examples to evaluate in a single forward pass.

        Returns:
            List of prediction dicts matching `predict_bias` per input.
        """
        results: List[Dict] = []
        if not texts:
            return results

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            for j in range(len(batch_texts)):
                bias_prob = probs[j, 1].item()
                results.append(
                    {
                        "bias_probability": bias_prob,
                        "classification": "BIASED" if bias_prob > 0.5 else "NEUTRAL",
                        "confidence": max(bias_prob, 1 - bias_prob),
                    }
                )

        return results
