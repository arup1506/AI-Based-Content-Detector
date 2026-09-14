"""
Stylometric and Linguistic Feature Extraction for AI Text Detection.
Implements:
- Burstiness (Sentence length variation and structural rhythm)
- Perplexity / Entropy approximation (N-gram surprisal proxy)
- Lexical Diversity (Type-Token Ratio, Hapax Legomena)
- Formulaic LLM Discourse Marker Analysis
- Readability metrics (Flesch Reading Ease & Grade Level)
"""

import math
import re
from typing import List, Dict, Any, Tuple
from collections import Counter


# Common formulaic phrases heavily over-indexed in LLM outputs
LLM_MARKERS = [
    r"\bin conclusion\b",
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bit is important to remember\b",
    r"\bit is worth noting that\b",
    r"\bdelve into\b",
    r"\bdelves into\b",
    r"\btapestry of\b",
    r"\btestament to\b",
    r"\bbeacon of\b",
    r"\bin today's fast-paced world\b",
    r"\bnavigating the complexities\b",
    r"\bplays a crucial role\b",
    r"\bfoster a sense of\b",
    r"\bat its core\b",
    r"\bin summary\b",
    r"\bnot only .*? but also\b",
    r"\bunveils\b",
    r"\bseamlessly integrates\b",
    r"\bvital component\b",
    r"\bparamount importance\b",
    r"\bunderscores the need\b"
]


def split_sentences(text: str) -> List[str]:
    """
    Robust sentence boundary detection without external download dependencies.
    Handles abbreviations, quotes, and punctuation correctly.
    """
    if not text or not text.strip():
        return []
    
    # Normalize clean single lines within paragraphs
    text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
    
    # Split by standard sentence terminators (. ! ?) followed by whitespace or quote
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\“\‘])', text)
    
    cleaned_sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) > 1:
            cleaned_sentences.append(s)
            
    return cleaned_sentences if cleaned_sentences else [text.strip()]


def tokenize_words(text: str) -> List[str]:
    """Extract clean lowercase alphabetic words."""
    return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())


def count_syllables(word: str) -> int:
    """Heuristic syllable counter for English words."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    # Remove silent e
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    # Count vowel groups
    syllables = len(re.findall(r'[aeiouy]{1,2}', word))
    return max(1, syllables)


def calculate_burstiness(sentences: List[str]) -> Tuple[float, float, float]:
    """
    Calculate sentence length variation (Burstiness).
    Returns (mean_length, std_dev, coefficient_of_variation).
    AI text typically has low variation; human text has high burstiness.
    """
    if not sentences:
        return 0.0, 0.0, 0.0
    
    lengths = [len(tokenize_words(s)) for s in sentences if len(tokenize_words(s)) > 0]
    if not lengths:
        return 0.0, 0.0, 0.0
    
    mean_len = sum(lengths) / len(lengths)
    if len(lengths) < 2:
        return float(mean_len), 0.0, 0.0
        
    variance = sum((l - mean_len) ** 2 for l in lengths) / (len(lengths) - 1)
    std_dev = math.sqrt(variance)
    cv = (std_dev / mean_len) if mean_len > 0 else 0.0
    
    return float(mean_len), float(std_dev), float(cv)


def calculate_lexical_diversity(words: List[str]) -> Dict[str, float]:
    """
    Calculate vocabulary richness:
    - Type-Token Ratio (TTR)
    - Hapax Legomena Ratio (words appearing exactly once)
    """
    if not words:
        return {"ttr": 0.0, "hapax_ratio": 0.0, "unique_words": 0}
    
    total = len(words)
    counts = Counter(words)
    unique = len(counts)
    hapax = sum(1 for w, c in counts.items() if c == 1)
    
    return {
        "ttr": round(unique / total, 4),
        "hapax_ratio": round(hapax / total, 4),
        "unique_words": unique
    }


def calculate_entropy_proxy(text: str) -> float:
    """
    Calculate Shannon character bigram entropy as a proxy for perplexity / predictability.
    Lower entropy indicates more predictable, standard model-generated text.
    """
    clean = re.sub(r'\s+', ' ', text.lower().strip())
    if len(clean) < 10:
        return 0.0
        
    bigrams = [clean[i:i+2] for i in range(len(clean) - 1)]
    if not bigrams:
        return 0.0
        
    total_bigrams = len(bigrams)
    counts = Counter(bigrams)
    
    entropy = 0.0
    for count in counts.values():
        p = count / total_bigrams
        entropy -= p * math.log2(p)
        
    return round(entropy, 4)


def calculate_readability(words: List[str], sentences: List[str]) -> Dict[str, float]:
    """
    Calculate Flesch Reading Ease and Flesch-Kincaid Grade Level.
    """
    num_words = max(1, len(words))
    num_sentences = max(1, len(sentences))
    num_syllables = sum(count_syllables(w) for w in words)
    
    # Flesch Reading Ease
    asl = num_words / num_sentences  # Average Sentence Length
    asw = num_syllables / num_words  # Average Syllables per Word
    
    fre = 206.835 - (1.015 * asl) - (84.6 * asw)
    fre = max(0.0, min(100.0, round(fre, 2)))
    
    # Flesch-Kincaid Grade Level
    fkgl = (0.39 * asl) + (11.8 * asw) - 15.59
    fkgl = max(1.0, min(20.0, round(fkgl, 1)))
    
    return {
        "reading_ease": fre,
        "grade_level": fkgl,
        "avg_sentence_length": round(asl, 1),
        "avg_syllables_per_word": round(asw, 2)
    }


def detect_llm_markers(text: str) -> List[str]:
    """Find matches of typical LLM discourse markers."""
    matches = []
    text_lower = text.lower()
    for pattern in LLM_MARKERS:
        found = re.findall(pattern, text_lower)
        if found:
            matches.extend(found)
    return list(set(matches))


def extract_all_features(text: str) -> Dict[str, Any]:
    """
    Aggregate all stylometric, statistical, and linguistic features into a comprehensive dictionary.
    """
    sentences = split_sentences(text)
    words = tokenize_words(text)
    
    mean_len, std_dev, cv = calculate_burstiness(sentences)
    lex = calculate_lexical_diversity(words)
    entropy = calculate_entropy_proxy(text)
    readability = calculate_readability(words, sentences)
    markers = detect_llm_markers(text)
    
    return {
        "sentence_count": len(sentences),
        "word_count": len(words),
        "char_count": len(text),
        "burstiness_mean_len": round(mean_len, 2),
        "burstiness_std_dev": round(std_dev, 2),
        "burstiness_cv": round(cv, 3),
        "lexical_ttr": lex["ttr"],
        "lexical_hapax": lex["hapax_ratio"],
        "unique_words": lex["unique_words"],
        "entropy": entropy,
        "reading_ease": readability["reading_ease"],
        "grade_level": readability["grade_level"],
        "avg_sentence_len": readability["avg_sentence_length"],
        "llm_markers_found": markers,
        "llm_marker_count": len(markers),
        "sentences": sentences
    }
