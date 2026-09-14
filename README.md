# 🛡️ AI Content Detector (AI vs. Human Text)

An AI-generated text detection application that analyzes content from:
1. 📝 **Direct Text Input** (Copy/Paste with instant sample loaders)
2. 📁 **Document Files** (PDF `.pdf`, Word `.docx`, Plain text `.txt`, `.md`, `.rtf`)
3. 🔗 **Article & Blog URLs** (Automatic web scraper extracting clean article body)

Outputs comprehensive **AI % vs. Human %** distributions, overall verdict, confidence rating, sentence-by-sentence visual heatmaps, and stylometric diagnostics (burstiness, lexical diversity, readability).

---

## 🌟 Key Features

- **Percentage Breakdown**: Exact probability calculation (e.g., `85.2% AI-Generated, 14.8% Human-Authored`).
- **Multi-Document Upload**: Fast parser for `.pdf` (using `pypdf`), `.docx` (using `python-docx` + zero-dependency XML fallback), and text files.
- **Article & Blog URL Scraper**: Extracts clean editorial text while eliminating ads, navigation, headers, and footers.
- **Sentence-Level AI Heatmap**: Color-codes every sentence in the document based on individual model predictability:
  - 🔴 **Red**: Likely AI-Generated (> 65%)
  - 🟡 **Yellow**: Mixed / Uncertain (38% - 65%)
  - 🟢 **Green**: Likely Human-Authored (< 38%)
- **Stylometric & Linguistic Diagnostics**:
  - **Burstiness (CV)**: Measures sentence length variation and cadence.
  - **Lexical Diversity (TTR)**: Type-Token Ratio and unique word counts.
  - **Readability**: Flesch Reading Ease score and grade level.
  - **Formulaic LLM Marker Detection**: Identifies over-indexed transition phrases.
- **Minimizes False Positives**: Incorporates academic calibration to protect human writers.
- **Exportable Reports**: One-click download of JSON analysis reports.

---

## 📂 Project Structure

```
project ABCD/
│
├── data/
│   ├── dataset.csv                  # Labeled training dataset (Human vs. AI)
│   └── generate_dataset.py          # Dataset generator and synthesizer
│
├── models/
│   └── ai_detector.joblib           # Trained model artifact (TF-IDF + Calibrated Classifier)
│
├── src/
│   ├── __init__.py
│   ├── extractors.py                # Text extraction from PDF, DOCX, TXT, and web links
│   ├── features.py                  # Burstiness, lexical diversity, readability, markers
│   └── model.py                     # Detector classifier, calibration, sentence-level analyzer
│
├── templates/
│   └── index.html                   # Modern glassmorphism HTML dashboard for Flask
├── static/
│   ├── css/style.css                # Polished styling, badges, cards, responsive layout
│   └── js/app.js                    # AJAX controllers, progress bars, sentence highlighter
│
├── app.py                           # Streamlit Interactive Web Application
├── flask_app.py                     # Flask Web Dashboard & REST API
├── train.py                         # Model training and evaluation script
├── test_detector.py                 # Unit tests for extractors, features, and model
├── test_flask.py                    # Unit tests for Flask API endpoints
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container deployment definition
├── run_app.bat                      # Windows one-click launcher
├── run_app.sh                       # Linux / macOS launcher
├── METHODOLOGY_AND_EVALUATION.md    # Documentation of methodology & limitations
└── README.md                        # Documentation and deployment guide
```

---

## 🚀 Quick Start (Local)

### Option 1: Streamlit Web App (Recommended for Demo)
Run directly from your terminal:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option 2: Flask Web App & REST API
Run the Flask server:
```bash
python flask_app.py
```
Open your browser at `http://localhost:5000`.

### Option 3: One-Click Launchers
- **Windows**: Double-click `run_app.bat`
- **Linux / macOS**: Run `./run_app.sh`

---

## 🔌 REST API Endpoints (Flask)

### 1. Analyze Plain Text
```bash
curl -X POST http://localhost:5000/api/detect \
     -H "Content-Type: application/json" \
     -d '{"text": "Your text to analyze..."}'
```

### 2. Upload and Analyze Document (PDF/Word/TXT)
```bash
curl -X POST http://localhost:5000/api/upload-file \
     -F "file=@sample_essay.pdf"
```

### 3. Extract and Analyze Article URL
```bash
curl -X POST http://localhost:5000/api/analyze-url \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/blog-post"}'
```

---

## 🧪 Running Tests & Retraining

### Run Test Suite
```bash
python -m unittest test_detector.py
python -m unittest test_flask.py
```

### Retrain Model Pipeline
```bash
python data/generate_dataset.py
python train.py
```

---

## ☁️ Deployment Guide

### Deploying with Docker
Build and run the Docker container:
```bash
# Build Docker image
docker build -t ai-content-detector .

# Run container (Streamlit on port 8501)
docker run -p 8501:8501 ai-content-detector
```

### Deploying to Render / Cloud
1. Connect your repository to [Render](https://render.com).
2. Create a new **Web Service**.
3. Set:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt && python train.py`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0` (or `python flask_app.py`)

### Deploying to Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Choose **Streamlit** SDK.
3. Push the repository files. It will automatically detect `requirements.txt` and `app.py`.
