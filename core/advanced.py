import re
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import shap
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


# =========================
# SHAP-BASED EXPLAINER (Advanced)
# =========================


class SHAPExplainer:
    def __init__(self, model_path: str = "."):
        print("🚀 Initializing advanced SHAP Explainer...")
        # Load model/tokenizer directly (keep existing path behavior)
        try:
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            print(f"✅ Loaded DistilBERT from: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model from {model_path}: {e}")
            print("🔄 Falling back to distilbert-base-uncased")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=2
            )
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            print("✅ Fallback model loaded")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        print(f"✅ Model loaded on: {self.device}")

        def model_predict(texts):
            if isinstance(texts, str):
                texts = [texts]
            elif hasattr(texts, "__iter__") and not isinstance(texts, str):
                texts = list(texts)

            texts = [str(t) for t in texts]

            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                return probs[:, 1].cpu().numpy()

        self.model_predict = model_predict

        try:
            masker = shap.maskers.Text(tokenizer=self.tokenizer)
            self.explainer = shap.Explainer(self.model_predict, masker=masker)
            print("✅ SHAP Explainer initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing SHAP: {e}")
            self.explainer = None

    def _get_bias_probability(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            return probs[0, 1].item()

    def get_shap_values(self, text: str, max_evals: int = 500) -> List[Tuple[str, float]]:
        if self.explainer is None:
            return self._fallback_analysis(text)
        try:
            shap_values = self.explainer([text], max_evals=max_evals)
            if len(shap_values.values.shape) == 3:
                biased_values = shap_values.values[0, :, 0]
            else:
                biased_values = shap_values.values[0, :]
            tokens = shap_values.data[0]
            combined_scores = self._combine_subword_scores(tokens, biased_values)
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

    def _combine_subword_scores(self, tokens: List[str], scores: np.ndarray) -> List[Tuple[str, float]]:
        combined = []
        current_word, current_score, token_count = "", 0.0, 0
        for token, score in zip(tokens, scores):
            if token.startswith("##"):
                current_word += token[2:]
                current_score += score
                token_count += 1
            else:
                if current_word:
                    combined.append((current_word, current_score / token_count))
                current_word, current_score, token_count = token, score, 1
        if current_word:
            combined.append((current_word, current_score / token_count))
        return [(word.replace("##", "").strip(), score) for word, score in combined]

    def _fallback_analysis(self, text: str) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            bias_prob = probs[0, 1].item()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_scores = []
        for token in tokens:
            if token in ["[cls]", "[sep]", "[pad]"]:
                continue
            clean_token = token.replace("##", "").strip()
            if len(clean_token) <= 1:
                continue
            score = len(clean_token) * 0.1
            if clean_token.lower() in ["women", "men", "female", "male", "she", "he"]:
                score += 0.5
            if clean_token.lower() in ["nurse", "engineer", "teacher", "secretary"]:
                score += 0.3
            if clean_token.lower() in ["emotional", "aggressive", "decisive", "leadership"]:
                score += 0.2
            word_scores.append((clean_token, score * bias_prob))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return word_scores[:8]

    def analyze_text(self, text: str) -> Dict[str, Any]:
        bias_prob = self._get_bias_probability(text)
        shap_results = self.get_shap_values(text)
        bias_label, bias_types = self._classify_bias(text, bias_prob, shap_results)
        return {
            "text": text,
            "bias_probability": bias_prob,
            "bias_label": bias_label,
            "bias_types": bias_types,
            "shap_explanations": shap_results,
            "top_biased_words": [w for w, s in shap_results[:3]],
        }

    def _classify_bias(self, text: str, bias_prob: float, shap_results: List[Tuple[str, float]]) -> Tuple[str, List[str]]:
        text_lower = text.lower()
        bias_types: List[str] = []
        rule_score = 0.0
        group_words = ["women", "men", "woman", "man", "she", "he", "female", "male"]
        role_words = ["nurse", "engineer", "secretary", "teacher", "leader", "leadership", "manager"]
        aggression_phrases = ["more aggressive", "be more aggressive"]
        has_group_word = any(g in text_lower for g in group_words)
        if has_group_word and any(r in text_lower for r in role_words):
            if any(p in text_lower for p in ["should be", "supposed to be", "meant to be"]):
                bias_types.append("gender_role_stereotype")
                rule_score = max(rule_score, 0.8)
        if has_group_word and any(p in text_lower for p in [
            "are better at", "naturally better", "inherently better", "better at"
        ]):
            bias_types.append("gender_superiority")
            rule_score = max(rule_score, 0.8)
        if "too emotional" in text_lower or "less emotional" in text_lower:
            bias_types.append("emotionality_judgment")
            rule_score = max(rule_score, 0.7)
        elif "emotional" in text_lower and has_group_word:
            bias_types.append("emotionality_judgment")
            rule_score = max(rule_score, 0.6)
        if any(phrase in text_lower for phrase in aggression_phrases):
            if "to succeed" in text_lower or "for success" in text_lower:
                bias_types.append("aggression_norm")
                rule_score = max(rule_score, 0.7)
        skill_based = (
            "people with" in text_lower and
            "skills" in text_lower and
            any(f in text_lower for f in ["role", "roles", "position", "positions", "job", "jobs"])
        )
        if (not has_group_word) and skill_based and not bias_types:
            effective_bias_prob = min(bias_prob, 0.2)
        else:
            effective_bias_prob = max(bias_prob, rule_score)
        if effective_bias_prob >= 0.7:
            label = "BIASED"
        elif effective_bias_prob >= 0.3:
            label = "POSSIBLY_BIASED"
        else:
            label = "LIKELY_NEUTRAL"
        return label, sorted(list(set(bias_types)))


# =========================
# GRAMMAR / REWRITE SUPPORT (Advanced)
# =========================


class GrammarPolisher:
    def __init__(self):
        print("🔧 Loading advanced grammar polisher (FLAN-T5)...")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("✅ Grammar polisher loaded!")
        self.issues = {
            r"\b(\w+) skills skills\b": r"\1 skills",
            r"\b(\w+) abilities abilities\b": r"\1 abilities",
            r"\bdemonstrate well-suited\b": "are well-suited",
            r"\bpeople may be healthcare\b": "people may pursue healthcare roles",
            r"\bat technical roles\b": "for technical roles",
            r"\bin technical\b": "in technical fields",
            r"\bcan benefit from being enhanced assertiveness\b": "can benefit from being more assertive",
            r"\bpeople demonstrate emotional intelligence for\b": "emotional intelligence can be valuable for",
            r"\bthey needs\b": "they need",
            r"\bpeople needs\b": "people need",
            r"\bthe the\b": "the",
        }

    def _t5_rewrite(self, instruction: str, text: str, max_length: int = 128) -> str:
        prompt = f"{instruction}\n\nText: {text}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return result

    def polish_phrase(self, text: str) -> str:
        for pattern, replacement in self.issues.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def fix_grammar_only(self, text: str) -> str:
        rewritten = self._t5_rewrite(
            "Rewrite the following sentence to fix grammar and clarity without changing its meaning.",
            text,
        )
        return self.polish_phrase(rewritten)

    def rewrite_bias_free(self, text: str) -> str:
        rewritten = self._t5_rewrite(
            "Rewrite the following sentence to remove any bias or stereotypes while preserving its core meaning and context, and fix any grammatical issues.",
            text,
        )
        return self.polish_phrase(rewritten)


FIELD_TO_SKILL = {
    "healthcare": "clinical and interpersonal skills",
    "technical": "technical expertise and problem-solving",
    "leadership": "decision-making and communication",
    "administrative": "organizational and communication skills",
    "business": "strategic and interpersonal skills",
    "professional": "relevant professional skills",
}


class PerfectCounterfactualGenerator:
    def __init__(self):
        print("🚀 Initializing Perfect Counterfactual Generator...")
        self.grammar_polisher = GrammarPolisher()
        print("✅ Perfect Counterfactual Generator initialized!")

    def _clean_sentence(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[-1] not in ".!?":
            text += "."
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text

    def _normalize_for_compare(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def _detect_field(self, text_lower: str) -> str:
        if "nurse" in text_lower or "nursing" in text_lower:
            return "healthcare"
        if "engineer" in text_lower or "engineering" in text_lower:
            return "technical"
        if "secretary" in text_lower or "administrative" in text_lower:
            return "administrative"
        if "teacher" in text_lower or "teaching" in text_lower:
            return "professional"
        if "leader" in text_lower or "leadership" in text_lower:
            return "leadership"
        if "business" in text_lower:
            return "business"
        return "professional"

    def _get_default_skill_for_field(self, field: str) -> str:
        return FIELD_TO_SKILL.get(field, "relevant professional skills")

    # Pattern-specific rewrites (subset for brevity; extend as needed)
    def _rewrite_gender_role_stereotype(self, text: str) -> List[str]:
        text_lower = text.lower()
        field = self._detect_field(text_lower)
        variants: List[str] = []
        trait = None
        m = re.search(r"because they are ([^\.]+)", text_lower)
        if m:
            trait_phrase = m.group(1).strip().rstrip(".")
            if "compassionate" in trait_phrase:
                trait = "interpersonal skills"
            elif "caring" in trait_phrase:
                trait = "supportive qualities"
            else:
                trait = trait_phrase
        field_skill = trait if trait else self._get_default_skill_for_field(field)
        v1 = f"Roles in {field} often benefit from strong {field_skill}."
        variants.append(self._clean_sentence(v1))
        v2 = f"People with strong {field_skill} can succeed in {field} roles, regardless of gender."
        variants.append(self._clean_sentence(v2))
        v3 = (
            f"Selection for {field} roles should be based on {field_skill}, experience, and qualifications, not on gender."
        )
        variants.append(self._clean_sentence(v3))
        return variants

    def _rewrite_gender_superiority(self, text: str) -> List[str]:
        text_lower = text.lower()
        field = self._detect_field(text_lower)
        skill = self._get_default_skill_for_field(field)
        variants: List[str] = []
        v1 = (
            f"Success in {field} roles depends on {skill}, training, and experience, rather than belonging to any particular gender group."
        )
        variants.append(self._clean_sentence(v1))
        v2 = f"People of any gender can excel in {field} roles when they develop strong {skill}."
        variants.append(self._clean_sentence(v2))
        v3 = f"Ability in {field} comes from learning and practice, not from being part of a specific gender."
        variants.append(self._clean_sentence(v3))
        return variants

    def _rewrite_emotionality_judgment(self, text: str) -> List[str]:
        text_lower = text.lower()
        variants: List[str] = []
        group_words = ["women", "men", "woman", "man", "she", "he", "female", "male"]
        is_group = any(text_lower.startswith(g) or f" {g} " in text_lower for g in group_words)
        if is_group:
            field = self._detect_field(text_lower)
            if "leadership" in text_lower:
                field = "leadership"
            skill = self._get_default_skill_for_field(field)
            v1 = f"{field.capitalize()} roles require managing emotions effectively and making balanced decisions."
            v2 = f"People of any gender can develop the {skill} needed to succeed in {field} positions."
            v3 = f"Evaluating candidates for {field} roles should focus on skills and behavior, not assumptions about gender."
            for v in [v1, v2, v3]:
                variants.append(self._clean_sentence(v))
            return variants
        if "needs to be less emotional" in text_lower:
            v1 = (
                "The candidate may benefit from managing emotional responses more effectively in professional settings."
            )
            v2 = (
                "Providing support for emotional self-management could help the candidate perform more consistently."
            )
            for v in [v1, v2]:
                variants.append(self._clean_sentence(v))
            return variants
        if "too emotional" in text_lower:
            v1 = (
                "It may be helpful to support the person in managing strong emotional reactions in demanding situations."
            )
            return [self._clean_sentence(v1)]
        return []

    def _rewrite_aggression_norm(self, text: str) -> List[str]:
        text_lower = text.lower()
        field = "business" if "business" in text_lower else "professional"
        replacement_trait = "more assertive"
        variants: List[str] = []
        draft1 = re.sub(
            r"needs to be more aggressive",
            f"may benefit from being {replacement_trait}",
            text,
            flags=re.IGNORECASE,
        )
        if draft1 != text:
            variants.append(self._clean_sentence(draft1))
        v2 = f"Success in {field} often benefits from being assertive and clear in communication."
        variants.append(self._clean_sentence(v2))
        v3 = (
            f"In {field} contexts, taking initiative and communicating confidently can support better outcomes."
        )
        variants.append(self._clean_sentence(v3))
        return variants

    def generate_counterfactuals(
        self,
        text: str,
        shap_results: List[Tuple[str, float]],
        bias_label: str,
        bias_types: List[str],
        num_alternatives: int = 3,
    ) -> List[str]:
        text_lower = text.lower()
        original_norm = self._normalize_for_compare(text)
        suggestions: List[str] = []
        if bias_label == "LIKELY_NEUTRAL" and not bias_types:
            grammar_fixed = self.grammar_polisher.fix_grammar_only(text)
            grammar_fixed = self._clean_sentence(grammar_fixed)
            gf_norm = self._normalize_for_compare(grammar_fixed)
            if grammar_fixed and gf_norm != original_norm:
                suggestions.append(grammar_fixed)
            bias_free = self.grammar_polisher.rewrite_bias_free(text)
            bias_free = self._clean_sentence(bias_free)
            bf_norm = self._normalize_for_compare(bias_free)
            if bias_free and bf_norm != original_norm and bias_free not in suggestions:
                suggestions.append(bias_free)
            if not suggestions:
                suggestions.append(self._clean_sentence(text))
            return suggestions[:num_alternatives]
        for bias_type in bias_types:
            variants: List[str] = []
            if bias_type == "gender_role_stereotype":
                variants = self._rewrite_gender_role_stereotype(text)
            elif bias_type == "gender_superiority":
                variants = self._rewrite_gender_superiority(text)
            elif bias_type == "emotionality_judgment":
                if "presentation" in text_lower and "emotional" in text_lower:
                    variants = [
                        self._clean_sentence(
                            re.sub(
                                r"was very emotional",
                                "showed a strong emotional reaction",
                                text,
                                flags=re.IGNORECASE,
                            )
                        ),
                        "During the presentation, the speaker displayed a strong emotional response.",
                    ]
                else:
                    variants = self._rewrite_emotionality_judgment(text)
            elif bias_type == "aggression_norm":
                variants = self._rewrite_aggression_norm(text)
            for draft in variants:
                polished = self.grammar_polisher.rewrite_bias_free(draft)
                polished = self._clean_sentence(polished)
                p_norm = self._normalize_for_compare(polished)
                if polished and p_norm != original_norm and polished not in suggestions:
                    suggestions.append(polished)
                if len(suggestions) >= num_alternatives:
                    break
            if len(suggestions) >= num_alternatives:
                break
        if len(suggestions) < num_alternatives and len(suggestions) < 2 and bias_label in ("POSSIBLY_BIASED", "BIASED"):
            bias_free = self.grammar_polisher.rewrite_bias_free(text)
            bias_free = self._clean_sentence(bias_free)
            bf_norm = self._normalize_for_compare(bias_free)
            if bias_free and bf_norm != original_norm and bias_free not in suggestions:
                suggestions.append(bias_free)
        if len(suggestions) < num_alternatives and "needs to be more aggressive" in text_lower and "business" in text_lower:
            direct = "He may benefit from being more assertive to succeed in business."
            direct = self._clean_sentence(direct)
            d_norm = self._normalize_for_compare(direct)
            if d_norm != original_norm and direct not in suggestions:
                suggestions.append(direct)
        attempts = 0
        while len(suggestions) < num_alternatives and attempts < 3:
            attempts += 1
            fallback = self.grammar_polisher.rewrite_bias_free(text)
            fallback = self._clean_sentence(fallback)
            f_norm = self._normalize_for_compare(fallback)
            if not fallback or f_norm == original_norm or fallback in suggestions:
                continue
            suggestions.append(fallback)
        unique_suggestions: List[str] = []
        for s in suggestions:
            if s and s not in unique_suggestions:
                unique_suggestions.append(s)
            if len(unique_suggestions) >= num_alternatives:
                break
        return unique_suggestions[:num_alternatives]


class BiasGuardProAdvanced:
    def __init__(self, model_path: str = "."):
        print("🚀 Initializing BiasGuard Pro Advanced...")
        t0 = time.perf_counter()
        self.shap_explainer = SHAPExplainer(model_path)
        self.cf_generator = PerfectCounterfactualGenerator()
        self._init_time = time.perf_counter() - t0
        print("✅ BiasGuard Pro Advanced initialized successfully!\n")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        t_start = time.perf_counter()
        shap_analysis = self.shap_explainer.analyze_text(text)
        timings["shap_analyze_text"] = time.perf_counter() - t_start
        shap_results = shap_analysis["shap_explanations"]
        bias_prob = shap_analysis["bias_probability"]
        bias_label = shap_analysis["bias_label"]
        bias_types = shap_analysis["bias_types"]
        t_start = time.perf_counter()
        counterfactuals = self.cf_generator.generate_counterfactuals(
            text,
            shap_results,
            bias_label=bias_label,
            bias_types=bias_types,
            num_alternatives=3,
        )
        timings["generate_counterfactuals"] = time.perf_counter() - t_start
        timings["analyze_text_total"] = time.perf_counter() - t0
        return {
            "text": text,
            "bias_probability": bias_prob,
            "bias_class": bias_label,  # map to existing UI key
            "confidence": max(bias_prob, 1 - bias_prob),
            # Prefer positively-contributing tokens (increase bias probability).
            # If fewer than 3 positive tokens exist, fill with the largest-magnitude negatives.
            "top_biased_words": self._select_top_positive_words(shap_results, top_k=3),
            "shap_scores": shap_results[:10],
            "counterfactuals": counterfactuals,
            "bias_types": bias_types,
            "timestamp": time.time(),
            "timings": timings,
        }

    def _select_top_positive_words(self, shap_results: List[Tuple[str, float]], top_k: int = 3) -> List[str]:
        # Only return tokens with positive SHAP contributions (tokens that
        # increase the model's bias probability). If fewer than `top_k`
        # positives exist, return the smaller list (do not pad with
        # negatives).
        positives = [w for w, s in shap_results if s > 0]
        return positives[:top_k]
