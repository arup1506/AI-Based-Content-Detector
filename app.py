"""
Streamlit Web Application for AI-Generated Text Detector.
Provides:
- Three Input Tabs: Text Copy/Paste, Document Upload (PDF/Word/TXT), Article/Blog URLs
- Percentage Breakdown: AI % vs Human %
- Confidence Score & Verdict
- Stylometric Diagnostics (Burstiness, Lexical Diversity, Readability, Markers)
- Visual Sentence-Level Heatmap
- Report Download
"""

import os
import sys

# Ensure repository root is in sys.path for Streamlit Cloud / Linux
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import streamlit as st

try:
    from src.extractors import extract_from_file, extract_from_url, clean_text
    from src.model import AIDetector, MODEL_PATH
except ModuleNotFoundError as e:
    st.error(
        f"⚠️ **Module Import Error**: `{e}`\n\n"
        "If you are deploying on **Streamlit Cloud**, please ensure:\n"
        "1. The **`src/`** folder (containing `extractors.py`, `features.py`, `model.py`) was pushed/uploaded to your GitHub repository.\n"
        "2. The **`requirements.txt`** file is in the root directory of your GitHub repository so Streamlit can install all dependencies."
    )
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="AI Content Detector - AI vs Human",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .main-subtitle {
        color: #6B7280;
        font-size: 1.05rem;
        max-width: 750px;
        margin: 0 auto;
    }
    
    /* Result Cards */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .card-ai {
        flex: 1;
        background: linear-gradient(145deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 1px solid #FECDD3;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .card-human {
        flex: 1;
        background: linear-gradient(145deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1px solid #BBF7D0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .pct-val {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .pct-ai { color: #E11D48; }
    .pct-human { color: #16A34A; }
    .pct-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #4B5563;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Verdict Banner */
    .verdict-box {
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1.2rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .verdict-ai {
        background-color: #FFF1F2;
        border-left: 6px solid #E11D48;
        color: #9F1239;
    }
    .verdict-human {
        background-color: #F0FDF4;
        border-left: 6px solid #16A34A;
        color: #14532D;
    }
    .verdict-mixed {
        background-color: #FFFBEB;
        border-left: 6px solid #D97706;
        color: #92400E;
    }
    
    /* Sentence Heatmap Highlights */
    .sentence-container {
        line-height: 1.85;
        font-size: 1.05rem;
        padding: 1.5rem;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .highlight-ai {
        background-color: #FECDD3;
        color: #9F1239;
        padding: 2px 5px;
        border-radius: 4px;
        border-bottom: 2px solid #E11D48;
        transition: all 0.2s ease;
    }
    .highlight-human {
        background-color: #DCFCE7;
        color: #14532D;
        padding: 2px 5px;
        border-radius: 4px;
        border-bottom: 2px solid #16A34A;
        transition: all 0.2s ease;
    }
    .highlight-mixed {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 2px 5px;
        border-radius: 4px;
        border-bottom: 2px solid #D97706;
        transition: all 0.2s ease;
    }
    
    /* Stat Pill */
    .stat-pill {
        display: inline-block;
        background: #F3F4F6;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #374151;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_detector() -> AIDetector:
    """Load and cache the detection model."""
    detector = AIDetector(model_path=MODEL_PATH)
    if not detector.is_trained:
        csv_file = os.path.join(ROOT_DIR, "data", "dataset.csv")
        if not os.path.exists(csv_file):
            try:
                from data.generate_dataset import save_dataset
                save_dataset(csv_file)
            except Exception:
                pass
        from train import train_and_evaluate
        train_and_evaluate()
        detector = AIDetector(model_path=MODEL_PATH)
    return detector


detector = get_detector()

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">AI Content Detector</div>
    <div class="main-subtitle">
        Deep linguistic & machine learning inspection to distinguish human-authored writing from AI models (ChatGPT, Claude, Gemini).
    </div>
</div>
""", unsafe_allow_html=True)

# Sample texts for quick demonstration
SAMPLE_TEXTS = {
    "AI (ChatGPT / GPT-4)": (
        "Artificial intelligence has emerged as a transformative paradigm across modern computing landscapes. "
        "By leveraging advanced deep neural networks and sophisticated machine learning algorithms, organizations can analyze vast "
        "troves of unstructured data with unprecedented efficiency. Furthermore, generative language models are revolutionizing natural "
        "language processing tasks, enabling automated content synthesis and intelligent conversational agents. However, it is important to remember "
        "that these systems also introduce substantial ethical considerations, including algorithmic bias, data privacy vulnerabilities, and explainability bottlenecks. "
        "Therefore, establishing comprehensive governance frameworks plays a crucial role in ensuring that technological innovation aligns with societal values. "
        "In conclusion, the ongoing integration of artificial intelligence will continue to reshape our technological horizon."
    ),
    "Human (Personal Essay)": (
        "Sitting in the drafty archives on a Tuesday morning, I kept wondering whether Spinoza ever foresaw the strange turns "
        "his substance monism would take in modern physics. The ink on the 1677 edition is faded, almost ghostly, but the geometric "
        "rigor still leaps off the vellum. It's easy to dismiss his God-or-Nature equation as mere pantheism, yet that misses the radical "
        "political bite of his Tractatus. I walked out into the biting drizzle around noon, grabbed a stale pretzel from the street vendor near "
        "the library steps, and watched pigeons fight over mustard packets. Philosophy rarely keeps your shoes dry, but it ruined my appetite in the best possible way."
    ),
    "Human (Technical Blog)": (
        "Spent three agonizing hours tracking down a memory leak in our WebSocket gateway yesterday. Turns out somebody registered a custom "
        "event listener inside a connection loop and completely forgot to detach it when clients dropped connection prematurely. In local tests "
        "with a dozen mock connections, everything looked squeaky clean. But when our staging cluster got hammered by our morning synthetic load test, "
        "the RSS footprint crept upward at roughly 80 megabytes per minute until Linux's OOM killer stepped in with extreme prejudice. "
        "Always verify teardown hooks in your mock harnesses, and never trust a garbage collector to clean up after your sloppy event emitters."
    )
}

# Input Section - 3 Tabs
tab_text, tab_file, tab_url = st.tabs([
    "📝 Paste Text", 
    "📁 Upload Document (PDF / Word / TXT)", 
    "🔗 Article / Blog Link"
])

text_to_analyze = ""
input_source_info = ""

# --- TAB 1: PASTE TEXT ---
with tab_text:
    st.markdown("#### Analyze Raw Text")
    col_sample1, col_sample2, col_sample3 = st.columns([1, 1, 1])
    
    selected_sample = None
    with col_sample1:
        if st.button("Load AI Sample (ChatGPT)"):
            st.session_state["raw_text_input"] = SAMPLE_TEXTS["AI (ChatGPT / GPT-4)"]
    with col_sample2:
        if st.button("Load Human Essay Sample"):
            st.session_state["raw_text_input"] = SAMPLE_TEXTS["Human (Personal Essay)"]
    with col_sample3:
        if st.button("Load Human Tech Blog"):
            st.session_state["raw_text_input"] = SAMPLE_TEXTS["Human (Technical Blog)"]

    raw_text = st.text_area(
        label="Enter or paste text (English, 50–2000+ words):",
        value=st.session_state.get("raw_text_input", ""),
        height=220,
        placeholder="Paste your essay, article, or writing sample here..."
    )
    
    words = len(raw_text.split()) if raw_text else 0
    chars = len(raw_text) if raw_text else 0
    st.caption(f"Word count: **{words}** | Character count: **{chars}** (Recommended: 80+ words)")
    
    if st.button("🚀 Analyze Pasted Text", type="primary", use_container_width=True):
        if words < 15:
            st.error("Please provide at least 15 words for meaningful detection.")
        else:
            text_to_analyze = raw_text
            input_source_info = "Pasted Text"

# --- TAB 2: UPLOAD DOCUMENT ---
with tab_file:
    st.markdown("#### Upload Document")
    st.markdown("Upload any **PDF (.pdf)**, **Word Document (.docx)**, or **Text file (.txt, .md)**.")
    
    uploaded_file = st.file_uploader(
        "Choose a document file", 
        type=["pdf", "docx", "txt", "md", "rtf", "csv"],
        help="Upload files up to 20MB. Encrypted or image-only scanned PDFs are not supported."
    )
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        
        try:
            extracted = extract_from_file(file_bytes, file_name)
            st.success(f"Successfully extracted {extracted['word_count']} words from **{file_name}** ({extracted['file_type']}).")
            
            with st.expander("📄 Document Preview (Click to expand)"):
                st.text_area("Extracted Content", extracted["raw_text"], height=200, disabled=True)
                
            if st.button("🚀 Analyze Uploaded Document", type="primary", use_container_width=True):
                if extracted["word_count"] < 15:
                    st.error("Extracted document has fewer than 15 words. Please upload a document with more text.")
                else:
                    text_to_analyze = extracted["raw_text"]
                    input_source_info = f"File: {file_name}"
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# --- TAB 3: ARTICLE / BLOG URL ---
with tab_url:
    st.markdown("#### Extract & Analyze Web Link")
    st.markdown("Provide the URL of any online blog post, news article, or publication.")
    
    input_url = st.text_input(
        "Article URL:",
        placeholder="https://example.com/article-slug"
    )
    
    if st.button("🌐 Fetch & Analyze Web Article", type="primary", use_container_width=True):
        if not input_url.strip():
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Fetching web page and extracting clean article body..."):
                try:
                    url_res = extract_from_url(input_url.strip())
                    st.success(f"Extracted **{url_res['word_count']} words** from: *{url_res['title']}*")
                    with st.expander("🌐 Article Preview"):
                        st.write(f"**Title**: {url_res['title']}")
                        st.text_area("Extracted Content", url_res["raw_text"], height=160, disabled=True)
                        
                    if url_res["word_count"] < 15:
                        st.warning("The extracted text from this URL is too short. It may be behind a paywall or heavy JavaScript.")
                    else:
                        text_to_analyze = url_res["raw_text"]
                        input_source_info = f"URL: {url_res['title']} ({input_url})"
                except Exception as e:
                    st.error(f"Failed to fetch content from URL: {str(e)}")


# --- RESULTS RENDERING ---
if text_to_analyze:
    st.divider()
    with st.spinner("Analyzing linguistic patterns, syntactic cadence, and statistical embeddings..."):
        try:
            results = detector.analyze(text_to_analyze)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()

    ai_pct = results["ai_percentage"]
    human_pct = results["human_percentage"]
    verdict = results["verdict"]
    confidence = results["confidence"]
    stats = results["stats"]

    # Header of Results
    st.markdown(f"### 📊 Detection Results & Breakdown")
    if input_source_info:
        st.caption(f"Source: **{input_source_info}** | {stats['total_words']} words | {stats['total_sentences']} sentences")

    # Percentage Cards
    col_ai, col_human = st.columns(2)
    with col_ai:
        st.markdown(f"""
        <div class="card-ai">
            <div class="pct-val pct-ai">{ai_pct}%</div>
            <div class="pct-label">AI-Generated Content</div>
            <div style="font-size:0.85rem; color:#881337; margin-top:0.4rem;">
                Predictable syntactic structure & typical LLM discourse
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_human:
        st.markdown(f"""
        <div class="card-human">
            <div class="pct-val pct-human">{human_pct}%</div>
            <div class="pct-label">Human-Authored Writing</div>
            <div style="font-size:0.85rem; color:#064E3B; margin-top:0.4rem;">
                Natural burstiness & idiosyncratic vocabulary cadence
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Progress bar visualizer
    st.progress(ai_pct / 100.0)

    # Verdict Box
    box_class = "verdict-ai" if ai_pct >= 65 else ("verdict-human" if human_pct >= 65 else "verdict-mixed")
    icon = "🤖" if ai_pct >= 65 else ("✍️" if human_pct >= 65 else "⚖️")
    st.markdown(f"""
    <div class="verdict-box {box_class}">
        <div>
            <div style="font-size:1.25rem; font-weight:700;">{icon} Final Verdict: {verdict}</div>
            <div style="font-size:0.92rem; margin-top:0.25rem;">{results['summary_description']}</div>
        </div>
        <div style="text-align:right;">
            <span style="background:rgba(0,0,0,0.08); padding:5px 12px; border-radius:20px; font-weight:600; font-size:0.88rem;">
                {confidence}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stylometric Metrics Grid
    st.markdown("#### 🔬 Stylometric & Linguistic Diagnostics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        cv = stats["burstiness_cv"]
        b_label = "High Variation (Human-like)" if cv > 0.50 else ("Uniform (AI-like)" if cv < 0.35 else "Moderate")
        st.metric(
            label="Burstiness (Sentence Variation)",
            value=f"{cv:.2f} CV",
            delta=b_label,
            delta_color="normal" if cv > 0.50 else "inverse"
        )
    with m2:
        ttr = stats["lexical_diversity"]
        st.metric(
            label="Lexical Diversity (TTR)",
            value=f"{ttr:.2f}",
            delta="Rich" if ttr > 0.65 else ("Standard" if ttr > 0.50 else "Repetitive")
        )
    with m3:
        fre = stats["reading_ease"]
        st.metric(
            label="Flesch Reading Ease",
            value=f"{fre:.1f} / 100",
            delta=f"Grade {stats['grade_level']}"
        )
    with m4:
        markers = results["linguistic_metrics"]["llm_markers_found"]
        st.metric(
            label="LLM Transition Markers",
            value=f"{len(markers)} Found",
            delta="Formulas Detected" if len(markers) > 0 else "None Detected",
            delta_color="inverse" if len(markers) > 0 else "normal"
        )

    if markers:
        st.caption(f"Detected formulaic markers: {', '.join([f'`{m}`' for m in markers])}")

    # Sentence-Level Heatmap
    st.markdown("#### 🔍 Sentence-by-Sentence AI Heatmap")
    st.caption("Sentences are color-coded based on individual model predictability:")
    st.markdown("""
    <div style="margin-bottom:0.75rem;">
        <span class="highlight-ai">Likely AI (&gt;65%)</span> &nbsp;
        <span class="highlight-mixed">Mixed / Uncertain (38% - 65%)</span> &nbsp;
        <span class="highlight-human">Likely Human (&lt;38%)</span>
    </div>
    """, unsafe_allow_html=True)

    highlighted_html_parts = []
    for s in results["sentences"]:
        score = s["ai_score"]
        tag = s["css_tag"]
        pct = s["ai_percentage"]
        # HTML badge highlight
        if tag == "ai-high":
            css = "highlight-ai"
        elif tag == "human-high":
            css = "highlight-human"
        else:
            css = "highlight-mixed"
            
        tooltip = f"AI Probability: {pct}%"
        highlighted_html_parts.append(
            f'<span class="{css}" title="{tooltip}">{s["text"]}</span>'
        )

    full_heatmap_html = (
        f'<div class="sentence-container">{" ".join(highlighted_html_parts)}</div>'
    )
    st.markdown(full_heatmap_html, unsafe_allow_html=True)

    # Sentence Details Table in Expander
    with st.expander("📋 Detailed Sentence Probability Table"):
        table_rows = []
        for s in results["sentences"]:
            table_rows.append({
                "#": s["index"],
                "Verdict": s["verdict"],
                "AI %": f"{s['ai_percentage']}%",
                "Human %": f"{s['human_percentage']}%",
                "Sentence Text": s["text"]
            })
        st.dataframe(table_rows, use_container_width=True)

    # Downloadable Report
    st.markdown("#### 📥 Export Detection Report")
    report_json = json.dumps(results, indent=2)
    st.download_button(
        label="Download Full JSON Analysis Report",
        data=report_json,
        file_name="ai_detection_report.json",
        mime="application/json"
    )
