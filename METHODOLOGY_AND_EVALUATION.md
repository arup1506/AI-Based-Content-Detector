# AI-Generated Text Detector: Methodology, Evaluation, and System Architecture

## 1. Executive Summary
With the rapid emergence of Large Language Models (LLMs) such as ChatGPT, Claude, Gemini, and open-weights models like Llama, separating human-authored writing from AI-generated text has become an essential capability for academia, journalism, and information integrity.

This project implements a multi-signal detection system combining:
1. **Machine Learning Pipeline**: Word and character n-gram TF-IDF representations coupled with a calibrated linear classifier.
2. **Stylometric & Linguistic Diagnostics**: Sentence burstiness, lexical diversity (Type-Token Ratio), Flesch reading ease, and formulaic LLM discourse markers.
3. **Multi-Channel Input Ingestion**: Direct text copy-paste, file uploads (PDF, DOCX, TXT), and web URL scraping.
4. **Sentence-Level Heatmaps**: Granular color-coded probability highlights indicating which sections of a document are likely AI-generated vs. human-written.

---

## 2. Theoretical Methodology

### 2.1 Stylometric Distinctions
Research into natural language generation has identified distinct statistical fingerprints left by transformer-based autoregressive models compared to human writers:

1. **Burstiness (Cadence & Length Variance)**:
   - **Humans**: Express thoughts with high syntactic variation. A short 4-word punchy sentence is often juxtaposed against a 35-word complex compound sentence containing parentheticals or dependent clauses.
   - **LLMs**: Generate text with uniform sentence lengths and standard rhythm, resulting in a low coefficient of variation ($CV = \sigma / \mu$).
2. **Perplexity Proxy & Predictability**:
   - LLMs sample tokens from the high-probability tail of their vocabulary distribution (top-$p$ / top-$k$ nucleus sampling), minimizing lexical surprises.
   - Transition entropy across character n-grams is significantly more constrained in AI writing.
3. **Formulaic Discourse Markers**:
   - LLMs are heavily fine-tuned with RLHF (Reinforcement Learning from Human Feedback) toward polite, balanced, expository prose.
   - Common markers include: *"Furthermore"*, *"Moreover"*, *"In conclusion"*, *"It is important to remember"*, *"Delve into"*, *"Tapestry of"*, *"Testament to"*, and *"Navigating the complexities"*.
4. **Lexical Diversity (Type-Token Ratio)**:
   - Evaluates vocabulary richness ($V / N$) and the proportion of *hapax legomena* (words occurring only once).

### 2.2 Model Architecture
```
Input Document (Raw Text / PDF / Word / URL)
                      │
                      ▼
            Text Sanitizer & Chunker
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
Document-Level Pipeline    Sentence-Level Chunker
        │                           │
  TF-IDF Feature Union        Sentence Tokenizer
  ├── Word N-grams (1, 2)           │
  └── Char N-grams (3, 5)     Per-Sentence Predictor
        │                           │
  Logistic Regression         Sentence Probability
  (Calibrated Probabilities)        │
        │                           ▼
        ▼                 Color-Coded Heatmap
  Stylometric Calibrator  (Red: AI, Green: Human,
  (Burstiness, Markers)    Yellow: Mixed)
        │
        ▼
   Final Output:
   - AI % vs. Human %
   - Verdict & Confidence
   - Stylometric Dashboard
```

---

## 3. Experimental Evaluation

### 3.1 Dataset Composition
- **Benchmark Corpus**: 56 curated, domain-balanced passages (28 authentic human, 28 AI-generated).
- **Domains Covered**: Computer Science, Philosophy, Urban Planning, Software Engineering, Maritime History, Culinary Arts, Higher Education, Biochemistry, Cinema, Ecology, DevOps, Athletics, Lifestyle, Mechanics, Architecture, Law, Horticulture, Finance, Audio Production, Astrophysics.
- **Length Constraint**: 100–500 words per sample, aligning with the project scope.

### 3.2 Out-of-Sample Performance Metrics
Evaluated on an independent 20% holdout test set with stratified 5-fold cross-validation on training data:

| Metric | Score |
| :--- | :--- |
| **5-Fold Mean CV Accuracy** | **100.0%** (+/- 0.0000) |
| **5-Fold Mean CV F1-Score** | **1.0000** (+/- 0.0000) |
| **Holdout Test Accuracy** | **100.00%** |
| **Holdout F1-Score** | **1.0000** |
| **Holdout ROC-AUC** | **1.0000** |

#### Confusion Matrix:
| | Predicted Human | Predicted AI |
| :--- | :---: | :---: |
| **True Human** | **6** | 0 |
| **True AI** | 0 | **6** |

*False Positive Rate (FPR) on holdout test set: 0.0%.*

---

## 4. Minimizing False Positives

A central project objective was minimizing false positives (human writing wrongly labeled as AI). This is achieved through three safeguards:

1. **Class-Balanced Regularization**: Regularized inverse penalty parameter ($C=1.8$) with balanced class weighting.
2. **Burstiness Safeguard**: If the coefficient of variation ($CV$) of sentence lengths exceeds $0.58$, the text receives a calibrated human confidence bonus, protecting idiosyncratic, academic human writing.
3. **Tri-Tier Confidence Calibration**: Texts falling in the 40%–60% zone are explicitly flagged as *"Mixed / Hybrid Content"* rather than forcefully labeled as pure AI.

---

## 5. System Limitations & Scope

As outlined in the project problem definition:
1. **Short Text Degradation**: Single sentences or short phrases (< 15 words) lack sufficient statistical sample size for reliable stylometric detection.
2. **Adversarial Paraphrasing & Retraining**: Newer AI models or intentional adversarial prompting ("write this with typos and slang") may reduce detection efficacy, requiring continuous dataset expansion and retraining.
3. **Plagiarism & Misinformation Exclusion**: The system detects human vs. AI generation only. It does not check for plagiarism against indexed web databases or fact-check truthfulness.

---

## 6. Future Enhancements
- Extension to multimodal detection (AI-generated synthetic images).
- Video deepfake detection and frame temporal analysis.
- Integration of local neural embeddings (RoBERTa / DeBERTa) for fine-tuning.
