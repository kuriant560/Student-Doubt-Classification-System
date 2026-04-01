import streamlit as st
import os
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from st_clipboard import copy_to_clipboard

import ollama

# Optional imports handled gracefully
try:
    import pdfplumber
    from docx import Document
    from pptx import Presentation
    EXTRACTION_READY = True
except ImportError:
    EXTRACTION_READY = False

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="Student Doubt Classification System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CORE LOGIC
# ================================
DATA_DIR = "data"
THRESHOLD = 0.15
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"

def chunk_text(text, chunk_size=500):
    """Character-based chunking."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\\n"
    return text

def extract_docx(file):
    doc = Document(file)
    return "\\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_pptx(file):
    prs = Presentation(file)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                text += shape.text + "\\n"
    return text

def extract_txt(file):
    return file.read().decode("utf-8")

@st.cache_data
def process_documents(uploaded_files):
    corpus, sources, errors, success_count = [], [], [], 0
    if uploaded_files:
        for file in uploaded_files:
            try:
                file.seek(0)
                ext = file.name.split(".")[-1].lower()
                text = ""
                
                if ext == "txt": text = extract_txt(file)
                elif ext == "pdf": text = extract_pdf(file)
                elif ext == "docx": text = extract_docx(file)
                elif ext == "pptx": text = extract_pptx(file)
                else:
                    errors.append(f"File type not supported: {file.name}")
                    continue
                
                if not text.strip():
                    errors.append(f"No text extracted from PDF (empty content): {file.name}")
                    continue
                
                chunks = chunk_text(text)
                if chunks:
                    corpus.extend(chunks)
                    sources.extend([file.name] * len(chunks))
                    success_count += 1
                else:
                    errors.append(f"No valid chunks generated from {file.name}")
            except Exception as e:
                errors.append(f"Could not extract from {file.name}: {e}")
    else:
        if os.path.exists(DATA_DIR):
            for filename in os.listdir(DATA_DIR):
                if filename.endswith(".txt"):
                    path = os.path.join(DATA_DIR, filename)
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                        chunks = chunk_text(text)
                        corpus.extend(chunks)
                        sources.extend([filename] * len(chunks))
    return corpus, sources, errors, success_count

def clean_output_text(text):
    text = text.replace("\\n", " ")
    text = re.sub(r'[^a-zA-Z0-9., ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def trim_text(text, max_len=400):
    return text[:max_len] + "..." if len(text) > max_len else text

def extract_key_sentences(text):
    sentences = text.split(".")
    return [s.strip() for s in sentences if s.strip()][:3]

def format_answer(text):
    sentences = extract_key_sentences(text)
    
    return f"""
Definition:
{sentences[0] + '.' if len(sentences) > 0 else ''}

Key Points:
- {sentences[1] + '.' if len(sentences) > 1 else ''}
- {sentences[2] + '.' if len(sentences) > 2 else ''}

Simple Explanation:
This concept explains how the system works in an easy and understandable way.
"""

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def expand_query(query):
    expansions = {
        "software design process": "software design process steps phases design methodology",
        "design process": "software design process phases steps",
    }
    for key in expansions:
        if key in query:
            return query + " " + expansions[key]
    return query


def get_best_match(query, corpus, sources):
    if not corpus: return None, 0.0, None, []
    
    processed_corpus = [clean_text(doc) for doc in corpus]
    expanded_query = expand_query(query)
    processed_query = clean_text(expanded_query)
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,3),
        max_df=0.9,
        min_df=1
    )
    
    tfidf_matrix = vectorizer.fit_transform(processed_corpus + [processed_query])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    best_idx = cosine_similarities.argmax()
    best_score = cosine_similarities[best_idx]
    
    top_indices = cosine_similarities.argsort()[-3:][::-1]
    top_matches = [(corpus[i], cosine_similarities[i], sources[i]) for i in top_indices if cosine_similarities[i] > 0]
    
    return corpus[best_idx], best_score, sources[best_idx], top_matches

def generate_fallback(query):
    """Intelligent fallback utilizing local Ollama engine."""
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant. Keep your answers extremely simple, concise, and easy for a beginner student to understand. Avoid long walls of text and overly complex terminology."},
                {"role": "user", "content": query}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error contacting local AI Fallback (Ollama): {str(e)}"


def ollama_response_failed(text):
    return isinstance(text, str) and text.startswith("Error contacting local AI Fallback (Ollama):")


def fallback_unavailable_message():
    return (
        "AI fallback is unavailable because Ollama is not running or not reachable on this machine. "
        f"Start Ollama locally and try again: {OLLAMA_DOWNLOAD_URL}"
    )

# Session State Init
if 'history' not in st.session_state: st.session_state.history = []

# ================================
# SIDEBAR UI
# ================================
st.sidebar.title("📁 Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload Knowledge Base files", 
    type=["txt", "pdf", "docx", "pptx"], 
    accept_multiple_files=True
)

st.sidebar.markdown("### Uploaded Documents")
if uploaded_files:
    for f in uploaded_files:
        st.sidebar.markdown(f"- {f.name}")
else:
    st.sidebar.info("Using default files from data/ folder.")

corpus, sources, errors, success_count = process_documents(uploaded_files)

if uploaded_files:
    if success_count > 0:
        st.sidebar.success(f"Successfully processed {success_count} files.")
    for err in errors:
        st.sidebar.warning(err)
    if not EXTRACTION_READY:
        st.sidebar.error("Warning: Advanced parsers missing. Run `pip install pdfplumber python-docx python-pptx`")


st.sidebar.markdown("### Detected Subjects")
unique_subjects = sorted(list(set([os.path.splitext(s)[0].replace("_", " ") for s in sources])))
if unique_subjects:
    for sub in unique_subjects:
        st.sidebar.markdown(f"- {sub}")
else:
    st.sidebar.markdown("- General")

# ================================
# MAIN SCREEN UI
# ================================
st.title("Student Doubt Classification System")
st.markdown("### Hybrid NLP (Document Retrieval + AI Fallback)")

# System Flow Visual
st.markdown("`Input → Preprocessing → TF-IDF → Similarity → Decision → Output`")
st.divider()

# Input
query = st.text_area("Ask your academic doubt...", placeholder="Enter question here...")

if st.button("Get Answer", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        st.session_state.history.append({"query": query, "timestamp": datetime.now().strftime("%H:%M:%S")})
        
        with st.spinner("Processing using NLP..."):
            
            # Step 5 Validation
            if len(corpus) == 0:
                st.error("No valid documents extracted. Cannot compute similarity. Please upload better documents.")
                st.stop()
                
            best_doc, confidence, source_file, top_matches = get_best_match(query, corpus, sources)
            
            is_fallback = confidence < THRESHOLD
            
            # Hybrid logic branches
            if is_fallback:
                subject = "GENERAL"
                with st.spinner("Generating AI response..."):
                    decision_label = "Switched to AI Fallback"
                    source_tag = "🤖 AI Generated Answer"
                    answer_text = generate_fallback(query)
            else:
                subject = os.path.splitext(source_file)[0].replace("_", " ").upper()
                decision_label = "Matched via Document Retrieval"
                source_tag = "📄 Answer from Documents"
                
                with st.spinner("Preparing answer from matched document..."):
                    cleaned = clean_output_text(best_doc)
                    trimmed = trim_text(cleaned)
                    
                    prompt = f"""
Convert the following academic content into a structured student answer with:
1. Definition
2. 3 bullet points
3. Simple explanation

Text:
{trimmed}
"""
                    try:
                        ai_rewritten_answer = generate_fallback(prompt)
                        if ollama_response_failed(ai_rewritten_answer):
                            answer_text = format_answer(trimmed)
                        else:
                            answer_text = ai_rewritten_answer
                    except Exception:
                        answer_text = format_answer(trimmed)

        # --------------------------
        # UI Rendering
        # --------------------------
        st.subheader("Results")
        st.markdown(f"**Your Question**: {query}")
        
        # Display Subject Badge
        st.markdown(f"**Subject**: `{subject}`")
        if not is_fallback:
            st.markdown(f"**Confidence**: `{round(confidence * 100, 2)}%`")
        else:
            st.markdown(f"**Confidence**: `AI Fallback`")
        
        # DEBUG OUTPUT
        with st.expander("🛠️ Advanced Debug Info"):
            st.info(f"""
            **NLP Pipeline Debug Info:**
            - Total Chunks Mapped: `{len(corpus)}`
            - Highest Similarity Score: `{confidence:.4f}`
            - Similarity Threshold: `{THRESHOLD:.2f}`
            """)
            
            if top_matches:
                for idx, (match_doc, match_score, match_source) in enumerate(top_matches):
                    st.markdown(f"**Match {idx+1}** | Score: `{match_score:.4f}` | Source: `{match_source}`\\n\\n> {match_doc[:300]}...")
            elif len(corpus) > 0:
                st.write(corpus[0] if len(corpus) > 0 else "Empty")
        
        if confidence > 0:
            st.progress(float(confidence))

        # Decision Output
        st.markdown("---")
        st.markdown(f"**Decision**: `{decision_label}`")
        st.markdown(f"**Source Tag**: *{source_tag}*")
        
        st.markdown("### 📘 Answer:")
        
        # Visual color cue for the answer output
        if is_fallback:
            if ollama_response_failed(answer_text):
                st.error(fallback_unavailable_message())
            else:
                st.info(answer_text)
            st.warning("Low confidence answer. Try rephrasing your query or upload better documents.")
        else:
            st.success(answer_text)
            
        # Try clipboard standard
        try:
            copy_to_clipboard(answer_text, text="📋 Copy", key=f"copy_{len(st.session_state.history)}")
        except Exception:
            pass
