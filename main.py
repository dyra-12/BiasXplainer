from core.bias_detector import BiasDetector
from core.explainer import SHAPExplainer
from core.counterfactuals import CounterfactualGenerator
from typing import Dict, List

class BiasGuardPro:
    def __init__(self, model_path: str = '.'):
        print("🚀 Initializing BiasGuard Pro...")
        self.detector = BiasDetector(model_path)
        self.explainer = SHAPExplainer(model_path)
        self.counterfactual_generator = CounterfactualGenerator()
        print("✅ BiasGuard Pro initialized successfully!\n")
    
    def analyze_text(self, text: str) -> Dict:
        """Complete analysis of text for bias"""
        print(f"🔍 Analyzing: '{text}'")
        
        # Get bias prediction
        bias_result = self.detector.predict_bias(text)
        
        # Get SHAP explanations
        shap_results = self.explainer.get_shap_values(text)
        print(f"   Top biased words: {[w for w, s in shap_results[:3]]}")
        
        # Generate counterfactuals
        counterfactuals = self.counterfactual_generator.generate_counterfactuals(text, shap_results)
        
        return {
            'text': text,
            'bias_probability': bias_result['bias_probability'],
            'bias_class': bias_result['classification'],
            'confidence': bias_result['confidence'],
            'top_biased_words': [w for w, s in shap_results[:3]],
            'shap_scores': shap_results[:10],
            'counterfactuals': counterfactuals
        }

def main():
    # Test the complete system
    bias_guard = BiasGuardPro(model_path='./')
    
    test_cases = [
        "Women should be nurses because they are compassionate.",
        "Men are naturally better at engineering roles.",
        "The female secretary was very emotional today.",
    ]
    
    print("🧪 Testing BiasGuard Pro System")
    print("=" * 50)
    
    for text in test_cases:
        result = bias_guard.analyze_text(text)
        
        print(f"\n📝 Original: {result['text']}")
        print(f"🎯 Bias Probability: {result['bias_probability']:.3f} ({result['bias_class']})")
        print(f"🔍 Top Biased Words: {result['top_biased_words']}")
        print("🔄 Counterfactuals:")
        for i, cf in enumerate(result['counterfactuals'], 1):
            print(f"   {i}. {cf}")
        print("-" * 50)

if __name__ == "__main__":
    main()