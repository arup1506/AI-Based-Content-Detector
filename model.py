"""
AI-Generated Text Detection Model & Scoring Engine.
Combines:
- Word and Character TF-IDF Feature Unions
- Calibrated Supervised Classification
- Stylometric & Heuristic Calibration (Burstiness, LLM Markers, Lexical Diversity)
- Sentence-by-sentence granularity for visual heatmaps
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

try:
    from src.features import extract_all_features, split_sentences, LLM_MARKERS
except ModuleNotFoundError:
    from features import extract_all_features, split_sentences, LLM_MARKERS

# Flexible model path resolution (works in root or in src/ / models/)
_curr = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.path.join(_curr, "models", "ai_detector.joblib"),
    os.path.join(_curr, "..", "models", "ai_detector.joblib"),
    os.path.join(_curr, "ai_detector.joblib"),
    os.path.join(os.getcwd(), "models", "ai_detector.joblib"),
    os.path.join(os.getcwd(), "ai_detector.joblib")
]
MODEL_PATH = next((p for p in _candidates if os.path.exists(p)), _candidates[0])


class AIDetector:
    """Production-grade AI vs. Human Text Detection Engine."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.is_trained = False
        self.load_or_initialize()

    def build_pipeline(self) -> Pipeline:
        """Construct a dual-granularity TF-IDF Pipeline with Logistic Regression."""
        union = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=2500,
                sublinear_tf=True,
                token_pattern=r'(?u)\b\w+\b'
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                max_features=3500,
                sublinear_tf=True
            ))
        ])

        clf = LogisticRegression(
            C=1.8,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )

        return Pipeline([
            ("features", union),
            ("classifier", clf)
        ])

    def fit(self, texts: List[str], labels: List[int]):
        """Train the classifier on a labeled text corpus."""
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(texts, labels)
        self.is_trained = True

    def save(self, target_path: str = None):
        """Save the trained model artifact."""
        target = target_path or self.model_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        joblib.dump(self.pipeline, target)
        print(f"Model successfully saved to: {target}")

    def load_or_initialize(self):
        """Load an existing model if found, otherwise ready to train."""
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                self.is_trained = True
            except Exception as e:
                print(f"Notice: Could not load existing model from {self.model_path}: {e}")
                self.pipeline = None
                self.is_trained = False
        else:
            self.pipeline = None
            self.is_trained = False

    def predict_sentence_score(self, sentence: str) -> float:
        """
        Evaluate probability of a single sentence being AI-generated (0.0 to 1.0).
        Blends model prediction with local heuristics.
        """
        words = sentence.split()
        if len(words) < 3:
            return 0.5  # Neutral for short fragments

        # Base model prediction
        if self.is_trained and self.pipeline:
            try:
                prob_ai = float(self.pipeline.predict_proba([sentence])[0][1])
            except Exception:
                prob_ai = 0.5
        else:
            prob_ai = 0.5

        # Heuristic boost for explicit LLM marker phrases in this sentence
        sentence_lower = sentence.lower()
        for marker in LLM_MARKERS:
            import re
            if re.search(marker, sentence_lower):
                prob_ai = min(1.0, prob_ai + 0.22)
                break

        return round(prob_ai, 3)

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of input text:
        - Global AI % and Human %
        - Stylometric feature breakdown
        - Sentence-level heatmap and classification
        - Confidence rating and explanatory summary
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty. Please provide text to analyze.")

        clean_content = text.strip()
        linguistic = extract_all_features(clean_content)
        sentences = linguistic["sentences"]
        word_count = linguistic["word_count"]

        if word_count < 10:
            raise ValueError(f"Text too short ({word_count} words). Please provide at least 15 words for reliable detection.")

        # 1. Base ML model probability
        if self.is_trained and self.pipeline:
            try:
                base_ai_prob = float(self.pipeline.predict_proba([clean_content])[0][1])
            except Exception:
                base_ai_prob = 0.5
        else:
            base_ai_prob = 0.5

        # 2. Stylometric Adjustments
        # AI characteristics: Low burstiness (cv < 0.35), high marker count, lower lexical diversity
        # Human characteristics: High burstiness (cv > 0.55), zero markers, natural cadence
        cv = linguistic["burstiness_cv"]
        marker_count = linguistic["llm_marker_count"]
        ttr = linguistic["lexical_ttr"]

        stylometric_delta = 0.0

        # Burstiness penalty / bonus
        if cv < 0.30 and len(sentences) >= 3:
            stylometric_delta += 0.10  # Very uniform -> lean AI
        elif cv > 0.58 and len(sentences) >= 3:
            stylometric_delta -= 0.12  # Very bursty -> lean Human

        # Marker influence
        if marker_count >= 2:
            stylometric_delta += min(0.20, marker_count * 0.08)
        elif marker_count == 0 and cv > 0.45:
            stylometric_delta -= 0.05  # Natural flow without formulaic buzzwords

        # Lexical diversity influence
        if ttr < 0.50 and word_count > 100:
            stylometric_delta += 0.05
        elif ttr > 0.75 and word_count > 100:
            stylometric_delta -= 0.05

        # Combine and clamp calibrated probability
        calibrated_ai = max(0.02, min(0.98, base_ai_prob + stylometric_delta))
        ai_percentage = round(calibrated_ai * 100, 1)
        human_percentage = round(100.0 - ai_percentage, 1)

        # 3. Sentence-level analysis for visual heatmap
        sentence_results = []
        for idx, s in enumerate(sentences):
            s_score = self.predict_sentence_score(s)
            s_ai_pct = round(s_score * 100, 1)
            
            if s_score >= 0.65:
                verdict = "AI-Generated"
                css_tag = "ai-high"
            elif s_score <= 0.38:
                verdict = "Human-Written"
                css_tag = "human-high"
            else:
                verdict = "Mixed / Uncertain"
                css_tag = "mixed"

            sentence_results.append({
                "index": idx + 1,
                "text": s,
                "ai_score": s_score,
                "ai_percentage": s_ai_pct,
                "human_percentage": round(100.0 - s_ai_pct, 1),
                "verdict": verdict,
                "css_tag": css_tag
            })

        # 4. Overall Verdict & Confidence
        if ai_percentage >= 68.0:
            verdict = "Likely AI-Generated"
            verdict_badge = "ai-verdict"
            summary_desc = (
                "The text exhibits high syntactic uniformity, consistent sentence lengths, "
                "and formulaic discourse markers strongly characteristic of LLM outputs (e.g., GPT, Claude, Gemini)."
            )
        elif human_percentage >= 68.0:
            verdict = "Likely Human-Authored"
            verdict_badge = "human-verdict"
            summary_desc = (
                "The text demonstrates natural burstiness, variable sentence cadence, "
                "and idiosyncratic vocabulary patterns typical of authentic human writing."
            )
        else:
            verdict = "Mixed / Hybrid Content"
            verdict_badge = "mixed-verdict"
            summary_desc = (
                "The content displays a combination of both AI-like structures and human-like variations, "
                "suggesting human editing of AI text or AI rephrasing of human draft."
            )

        # Confidence rating
        diff = abs(ai_percentage - 50.0)
        if diff >= 25.0:
            confidence = "High Confidence"
        elif diff >= 12.0:
            confidence = "Moderate Confidence"
        else:
            confidence = "Low / Borderline Confidence"

        return {
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "confidence": confidence,
            "ai_percentage": ai_percentage,
            "human_percentage": human_percentage,
            "summary_description": summary_desc,
            "linguistic_metrics": linguistic,
            "sentences": sentence_results,
            "stats": {
                "total_sentences": len(sentences),
                "total_words": word_count,
                "total_chars": linguistic["char_count"],
                "burstiness_cv": linguistic["burstiness_cv"],
                "reading_ease": linguistic["reading_ease"],
                "grade_level": linguistic["grade_level"],
                "lexical_diversity": linguistic["lexical_ttr"],
                "marker_count": linguistic["llm_marker_count"]
            }
        }
