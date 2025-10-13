import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import Dict, List

class BiasDetector:
    def __init__(self, model_path: str = '.'):
        print("Loading bias detection model...")
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()
        print(f"Bias detector loaded on: {self.device}")
    
    def predict_bias(self, text: str) -> Dict:
        """Predict bias probability for a single text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            bias_prob = probs[0, 1].item()
        
        return {
            'bias_probability': bias_prob,
            'classification': 'BIASED' if bias_prob > 0.5 else 'NEUTRAL',
            'confidence': max(bias_prob, 1 - bias_prob)
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict bias for multiple texts"""
        return [self.predict_bias(text) for text in texts]