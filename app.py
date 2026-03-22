"""
Scriptorium — Research Intelligence Platform
Professional AI chatbot for analyzing research papers with RAG technology.
"""

import os, time, datetime, tempfile, base64
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")

KEY = get_api_key()
if KEY:
    os.environ["OPENAI_API_KEY"] = KEY

MAX_Q = 10
MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
TOP_K = 4

st.set_page_config(
    page_title="Scriptorium",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL DARK THEME STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;600&family=Inter:wght@300;400;500&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #1a1a1a !important;
    color: #e0e0e0 !important;
}

.stMainBlockContainer {
    padding: 0 !important;
    max-width: 100% !important;
}

#MainMenu, footer, header { 
    visibility: hidden; 
}

/* ═════════════════════════════════════════ */
/* SIDEBAR */
/* ═════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background-color: #1f1f1f !important;
    border-right: 1px solid #333 !important;
}

section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}

.sidebar-logo {
    font-family: 'Crimson Text', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #f5f5f5;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #333;
}

.sidebar-section-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #888;
    margin: 1.5rem 0 0.8rem 0;
}

.sidebar-item {
    background: #2a2a2a;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.85rem;
}

.sidebar-item:hover {
    border-color: #555;
    background: #333;
}

/* ═════════════════════════════════════════ */
/* MAIN LAYOUT */
/* ═════════════════════════════════════════ */
.workspace-container {
    display: flex;
    height: 100vh;
    background: #1a1a1a;
    gap: 1px;
}

/* ═════════════════════════════════════════ */
/* PDF PANE */
/* ═════════════════════════════════════════ */
.pdf-pane {
    flex: 0 0 50%;
    display: flex;
    flex-direction: column;
    background: #2a2a2a;
    border-right: 1px solid #333;
    overflow: hidden;
}

.pdf-header {
    background: #1f1f1f;
    color: #e0e0e0;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #333;
    font-size: 0.85rem;
}

.pdf-header-line {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    color: #888;
}

.pdf-header-title {
    font-family: 'Crimson Text', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #f5f5f5;
    margin-bottom: 0.4rem;
}

.pdf-viewer {
    flex: 1;
    overflow-y: auto;
    background: #2a2a2a;
}

.pdf-viewer iframe {
    width: 100%;
    height: 100%;
    border: none;
}

/* ═════════════════════════════════════════ */
/* Q&A PANE */
/* ═════════════════════════════════════════ */
.qa-pane {
    flex: 0 0 50%;
    display: flex;
    flex-direction: column;
    background: #1a1a1a;
    overflow: hidden;
}

.qa-header {
    background: #1f1f1f;
    color: #e0e0e0;
    padding: 1.5rem;
    border-bottom: 1px solid #333;
}

.qa-header-title {
    font-family: 'Crimson Text', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #f5f5f5;
    margin-bottom: 0.3rem;
}

.qa-header-meta {
    font-size: 0.8rem;
    color: #888;
}

.qa-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.message-group {
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.message-question {
    color: #b0a89a;
    font-size: 0.9rem;
    margin-bottom: 0.8rem;
}

.message-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #d4af37;
    margin-bottom: 0.4rem;
    font-weight: 600;
}

.message-content {
    color: #c9b7a8;
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 0.6rem;
}

.key-points {
    margin-top: 0.8rem;
    padding-left: 1rem;
}

.key-point {
    color: #b0a89a;
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
    line-height: 1.5;
}

.evidence-box {
    background: rgba(212, 175, 55, 0.1);
    border-left: 3px solid #d4af37;
    padding: 0.8rem 1rem;
    margin-top: 0.8rem;
    border-radius: 4px;
}

.evidence-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #d4af37;
    margin-bottom: 0.3rem;
    font-weight: 600;
}

.evidence-text {
    color: #c9b7a8;
    font-size: 0.85rem;
    font-style: italic;
    line-height: 1.5;
}

/* ═════════════════════════════════════════ */
/* INPUT SECTION */
/* ═════════════════════════════════════════ */
.qa-input-section {
    padding: 1.5rem;
    border-top: 1px solid #333;
    background: #1f1f1f;
}

.suggested-prompts {
    margin-bottom: 1rem;
}

.prompt-chip {
    display: inline-block;
    padding: 0.6rem 1rem;
    background: #2a2a2a;
    color: #b0a89a;
    border: 1px solid #333;
    border-radius: 20px;
    font-size: 0.8rem;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.2s;
}

.prompt-chip:hover {
    background: #333;
    border-color: #555;
}

.qa-input-box {
    display: flex;
    gap: 0.8rem;
}

.qa-input {
    flex: 1;
    padding: 0.9rem 1.2rem;
    border: 1px solid #333;
    border-radius: 6px;
    font-family: 'Crimson Text', serif;
    font-size: 0.95rem;
    color: #e0e0e0;
    background: #2a2a2a;
}

.qa-input:focus {
    outline: none;
    border-color: #555;
    box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.1);
}

.qa-send-btn {
    padding: 0.9rem 1.5rem;
    background: #d4af37;
    color: #1a1a1a;
    border: none;
    border-radius: 6px;
    font-family: 'Crimson Text', serif;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.qa-send-btn:hover {
    background: #e6c957;
    transform: translateY(-1px);
}

/* ═════════════════════════════════════════ */
/* LANDING PAGE */
/* ═════════════════════════════════════════ */
.landing-container {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #1a1a1a;
    padding: 2rem;
}

.landing-box {
    text-align: center;
    max-width: 500px;
}

.landing-title {
    font-family: 'Crimson Text', serif;
    font-size: 2.5rem;
    font-weight: 600;
    color: #f5f5f5;
    margin-bottom: 0.5rem;
}

.landing-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #888;
    margin-bottom: 2rem;
}

.upload-zone {
    border: 2px dashed #333;
    border-radius: 8px;
    padding: 3rem 2rem;
    background: #2a2a2a;
    margin-bottom: 2rem;
    cursor: pointer;
    transition: all 0.2s;
}

.upload-zone:hover {
    border-color: #555;
    background: #333;
}

.upload-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.upload-text {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #b0a89a;
}

/* ═════════════════════════════════════════ */
/* SCROLLBAR */
/* ═════════════════════════════════════════ */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #2a2a2a;
}

::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #666;
}

/* ═════════════════════════════════════════ */
/* STREAMLIT OVERRIDES */
/* ═════════════════════════════════════════ */
.stButton > button {
    background-color: #2a2a2a !important;
    border: 1px solid #333 !important;
    color: #e0e0e0 !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background-color: #333 !important;
    border-color: #555 !important;
}

.stTextInput > div > div > input {
    background-color: #2a2a2a !important;
    color: #e0e0e0 !important;
    border: 1px solid #333 !important;
}

.stFileUploader {
    background-color: #2a2a2a !important;
    border: 1px dashed #333 !important;
    border-radius: 8px !important;
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════��════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "answers": [],
    "chain": None,
    "doc_name": None,
    "doc_stats": None,
    "pdf_b64": None,
    "q_count": 0,
    "pending_q": None,
    "thinking": False,
    "last_q": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">Scriptorium</div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="sidebar-section-title">Document</div>""", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    st.markdown("""<div class="sidebar-section-title">Navigation</div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Session", use_container_width=True):
            for k in ["answers", "chain", "doc_name", "doc_stats", "pdf_b64", "q_count"]:
                st.session_state[k] = [] if k == "answers" else (0 if k == "q_count" else None)
            st.rerun()

    with col2:
        if st.button("History", use_container_width=True):
            pass

    st.markdown("""<div class="sidebar-section-title">About</div>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.8rem; color: #888; line-height: 1.6;">
    Research Intelligence Platform for analyzing academic papers with precision.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_chain(_pdf_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_pdf_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.unlink(tmp_path)

    total = " ".join(p.page_content for p in pages).strip()
    if len(total) < 150:
        raise ValueError("SCANNED_PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        persist_directory=f"./chroma_{filename[:14].replace('.','_')}",
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3},
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer",
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a research assistant. Answer precisely from the context provided.

Context:
{context}

Question: {question}

Answer:""",
    )
    llm = ChatOpenAI(model_name=MODEL, temperature=0.1)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    stats = {
        "pages": len(pages),
        "chunks": len(chunks),
        "words": len(total.split()),
        "page_lengths": [len(p.page_content) for p in pages],
    }
    return chain, stats

def get_sources(source_docs):
    out, seen = [], set()
    for doc in source_docs:
        page = doc.metadata.get("page", 0)
        if page not in seen:
            seen.add(page)
            snippet = doc.page_content[:150].replace("\n", " ").strip()
            out.append({"page": page + 1, "snippet": snippet})
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# API KEY GUARD
# ═══════════════════════════════════════════════════════════════════════════════
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ No API key found. Add OPENAI_API_KEY to Streamlit secrets.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.chain is None:
    st.markdown("""
    <div class="landing-container">
        <div class="landing-box">
            <div class="landing-title">Scriptorium</div>
            <div class="landing-subtitle">Research Intelligence Platform</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if uploaded_file:
            try:
                pdf_bytes = uploaded_file.read()
                with st.spinner("Processing document..."):
                    chain, stats = build_chain(pdf_bytes, uploaded_file.name)
                st.session_state.chain = chain
                st.session_state.doc_name = uploaded_file.name
                st.session_state.doc_stats = stats
                st.session_state.pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.markdown("""
            <div class="upload-zone">
                <div class="upload-icon">📄</div>
                <div class="upload-text">Drop your PDF here or click to browse</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE - SPLIT SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""<div class="workspace-container">""", unsafe_allow_html=True)

    # LEFT PANE: PDF
    col_pdf, col_qa = st.columns([0.5, 0.5], gap="small")

    with col_pdf:
        st.markdown(f"""
        <div class="pdf-pane" style="height: 100vh; display: flex; flex-direction: column;">
            <div class="pdf-header">
                <div class="pdf-header-line">
                    <span>FileName: {st.session_state.doc_name}</span>
                    <span>[Zoom: 100%]</span>
                </div>
                <div class="pdf-header-line">
                    <span></span>
                    <span>[Pages: {st.session_state.doc_stats['pages']}]</span>
                </div>
            </div>
            <div class="pdf-viewer" style="flex: 1;">
                <iframe src="data:application/pdf;base64,{st.session_state.pdf_b64}" style="width:100%; height:100%; border:none;"></iframe>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # RIGHT PANE: Q&A
    with col_qa:
        st.markdown(f"""
        <div class="qa-pane" style="height: 100vh; display: flex; flex-direction: column;">
            <div class="qa-header">
                <div class="qa-header-title">Ask Your Questions</div>
                <div class="qa-header-meta">Meta text because 7 / 10 remaining</div>
            </div>
            
            <div class="qa-messages" style="flex: 1; overflow-y: auto;">
        """, unsafe_allow_html=True)

        # Display messages
        for item in st.session_state.answers:
            if not item.get("error"):
                st.markdown(f"""
                <div class="message-group">
                    <div class="message-question">[Question] {item['question']}</div>
                    <div class="message-label">Answer</div>
                    <div class="message-content">{item['content'][:400]}</div>
                    <div class="key-points">
                        <div class="key-point">• Key point from the paper</div>
                    </div>
                    <div class="evidence-box">
                        <div class="evidence-label">Evidence from Paper</div>
                        <div class="evidence-text">"{item.get('sources', [{'snippet': 'Evidence snippet'}])[0]['snippet']}"</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Input section
        if st.session_state.q_count < MAX_Q:
            st.markdown("""
            <div class="qa-input-section">
                <div class="suggested-prompts">
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Summarize this paper", use_container_width=True):
                    st.session_state.pending_q = "Summarize this paper"
                    st.session_state.q_count += 1
                    st.rerun()

            with col2:
                if st.button("What are key findings?", use_container_width=True):
                    st.session_state.pending_q = "What are the key findings?"
                    st.session_state.q_count += 1
                    st.rerun()

            with col3:
                if st.button("What methodology is used?", use_container_width=True):
                    st.session_state.pending_q = "What methodology is used?"
                    st.session_state.q_count += 1
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            user_input = st.text_input("Ask a question about the paper...", key="qa_input")

            if user_input:
                st.session_state.pending_q = user_input
                st.session_state.q_count += 1
                st.rerun()

            if st.session_state.pending_q and st.session_state.q_count > 0:
                try:
                    result = st.session_state.chain.invoke({"question": st.session_state.pending_q})
                    sources = get_sources(result.get("source_documents", []))
                    st.session_state.answers.append({
                        "question": st.session_state.pending_q,
                        "content": result["answer"],
                        "sources": sources,
                        "time": datetime.datetime.now().strftime("%H:%M"),
                        "error": False,
                    })
                except Exception as e:
                    st.session_state.answers.append({
                        "error": True,
                        "content": f"Error: {str(e)}"
                    })
                finally:
                    st.session_state.pending_q = None
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
