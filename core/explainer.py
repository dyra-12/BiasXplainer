import re
from typing import List, Tuple

import numpy as np
import shap
import torch

from .bias_detector import BiasDetector


class SHAPExplainer:
    def __init__(self, model_path: str = "."):
        """Initialize the SHAP-based explainer.

        Loads a BiasDetector from `model_path`, prepares a prediction wrapper
        suitable for SHAP (returns numpy arrays) and initializes the SHAP
        Explainer with a text masker. If SHAP initialization fails the
        explainer attribute will be set to None and a fallback path is
        available via `_fallback_analysis`.

        Args:
            model_path: Path or identifier used to load the underlying model/tokenizer.
        """
        print("Loading SHAP explainer...")
        self.detector = BiasDetector(model_path)

        def model_predict(texts):
            """Prediction wrapper used by SHAP.

            Converts incoming text(s) to a batched numpy array of shape (n,1)
            containing the model's bias probability. SHAP expects a callable
            that accepts a list/array of inputs and returns a numeric array.

            Args:
                texts: Single string or list of strings to score.

            Returns:
                numpy.ndarray: shape (n, 1) of bias probabilities.
            """
            if isinstance(texts, str):
                texts = [texts]
            texts = [str(t) for t in texts]

            # Use the bias detector's batched prediction for better throughput
            try:
                # Use a larger batch size to reduce per-call overhead during SHAP sampling
                preds = self.detector.predict_batch_batched(texts, batch_size=32)
                return np.array([[p.get("bias_probability", 0.0)] for p in preds])
            except Exception:
                # Fallback to single predictions if batching fails
                results = [self.detector.predict_bias(text) for text in texts]
                return np.array([[result["bias_probability"]] for result in results])

        self.model_predict = model_predict

        try:
            masker = shap.maskers.Text(tokenizer=self.detector.tokenizer)
            self.explainer = shap.Explainer(self.model_predict, masker=masker)
            print("✅ SHAP Explainer initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing SHAP: {e}")
            self.explainer = None

    def get_shap_values(
        self, text: str, max_evals: int = 500
    ) -> List[Tuple[str, float]]:
        """Compute SHAP contributions for words in `text`.

        If the SHAP explainer could not be created during init this will run a
        lightweight keyword-based fallback. The returned list contains tuples
        (word, score) sorted by absolute impact.

        Args:
            text: Input text to explain.
            max_evals: Maximum number of SHAP evaluations/samples.

        Returns:
            List of (word, score) tuples ordered by descending absolute impact.
        """
        if self.explainer is None:
            return self._fallback_analysis(text)

        try:
            shap_values = self.explainer([text], max_evals=max_evals)

            # SHAP returns different shaped arrays depending on explainer internals.
            # Normalize to a 1D array of token contributions.
            if len(shap_values.values.shape) == 3:
                biased_values = shap_values.values[0, :, 0]
            else:
                biased_values = shap_values.values[0, :]

            tokens = shap_values.data[0]

            # Combine subword tokens (e.g., BERT-style `##` tokens) into full words
            combined_scores = self._combine_subword_scores(tokens, biased_values)
            # Filter out special tokens and very short tokens
            filtered_scores = [
                (w, float(s))
                for w, s in combined_scores
                if w.strip() and w not in ["[cls]", "[sep]", "[pad]"] and len(w) > 1
            ]
            filtered_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            return filtered_scores[:10]

        except Exception as e:
            print(f"❌ SHAP calculation failed: {e}")
            return self._fallback_analysis(text)

    def _combine_subword_scores(
        self, tokens: List[str], scores: np.ndarray
    ) -> List[Tuple[str, float]]:
        """Merge BPE/subword tokens into human-readable words with averaged scores.

        Many tokenizers split words into subwords prefixed with '##'. This
        function recombines those tokens and averages their SHAP scores so the
        UI shows per-word importance instead of per-subword noise.

        Args:
            tokens: List of token strings from the tokenizer/SHAP.
            scores: Array of SHAP scores aligned with `tokens`.

        Returns:
            List of (word, averaged_score) tuples.
        """
        combined = []
        current_word, current_score, token_count = "", 0.0, 0

        for token, score in zip(tokens, scores):
            # If token is a continuation subword (BERT-style) start with '##'
            if token.startswith("##"):
                # append the subword (without the prefix) to the current word
                current_word += token[2:]
                current_score += score
                token_count += 1
            else:
                # push the previously accumulated word if present
                if current_word:
                    combined.append((current_word, current_score / token_count))
                current_word, current_score, token_count = token, score, 1

        if current_word:
            combined.append((current_word, current_score / token_count))

        # Final cleanup: remove any lingering '##' markers and strip whitespace
        return [(word.replace("##", "").strip(), score) for word, score in combined]

    def _fallback_analysis(self, text: str) -> List[Tuple[str, float]]:
        """Lightweight keyword-based analysis used when SHAP is unavailable.

        This function identifies a small set of bias-related keywords and
        assigns heuristic scores so the rest of the system can still show
        highlighted words and generate counterfactuals even without SHAP.

        Args:
            text: Input text to scan.

        Returns:
            List of (word, score) tuples sorted by score descending.
        """
        # Simple fallback based on known bias keywords
        bias_keywords = {
            "women",
            "men",
            "female",
            "male",
            "she",
            "he",
            "her",
            "his",
            "naturally",
            "should",
            "emotional",
            "aggressive",
            "compassionate",
            "nurse",
            "engineer",
            "secretary",
            "teacher",
            "better",
            "too",
        }

        words = re.findall(r"\b\w+\b", text.lower())
        scores = []

        for word in words:
            if word in bias_keywords:
                if word in ["women", "men", "naturally", "should"]:
                    scores.append((word, 0.8))
                elif word in ["emotional", "aggressive", "compassionate"]:
                    scores.append((word, 0.6))
                else:
                    scores.append((word, 0.4))

        return sorted(scores, key=lambda x: x[1], reverse=True)[:8]
