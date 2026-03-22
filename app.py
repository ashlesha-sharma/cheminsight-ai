"""
Scriptorium — research intelligence for any subject.
Professional UI with old money aesthetic, sidebar, search, filters & dark mode.
Built by Ashlesha Sharma, Miranda House, Delhi University.
"""

import os, time, datetime, tempfile, base64
import streamlit as st
from dotenv import load_dotenv
import plotly.graph_objects as go

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

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTES - OLD MONEY AESTHETIC
# ═══════════════════════════════════════════════════════════════════════════════
LIGHT_MODE = {
    "cream": "#F5F1E8",
    "burgundy": "#2C1810",
    "gold": "#D4AF37",
    "light_gold": "#E6C957",
    "off_white": "#FEFAF0",
    "shadow": "#1a1a1a",
    "burgundy_light": "#3D2817",
    "border": "#E0D5C5",
}

DARK_MODE = {
    "cream": "#1A1410",          # Dark burgundy-tinted background
    "burgundy": "#F5F1E8",       # Light text
    "gold": "#D4AF37",           # Same gold
    "light_gold": "#E6C957",     # Same light gold
    "off_white": "#2A1F18",      # Darker cards
    "shadow": "#0F0A08",         # Very dark
    "burgundy_light": "#3D2817", # Slightly lighter burgundy
    "border": "#4A3A2A",         # Dark border
}

st.set_page_config(
    page_title="Scriptorium",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "filter_by_relevance" not in st.session_state:
    st.session_state.filter_by_relevance = 0.0

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
    "documents": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# THEME SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════
THEME = DARK_MODE if st.session_state.dark_mode else LIGHT_MODE

# ══════���════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE STYLING WITH DARK MODE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600&family=Crimson+Text:wght@400;600&family=Inter:wght@300;400;500&display=swap');

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

html, body, [class*="css"] {{
    font-family: 'Crimson Text', 'Georgia', serif !important;
    background-color: {THEME['cream']} !important;
    color: {THEME['burgundy']} !important;
}}

.stMainBlockContainer {{
    padding: 0 !important;
    max-width: 100% !important;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

/* ═════════════════════════════════════════ */
/* SIDEBAR STYLING */
/* ═════════════════════════════════════════ */
section[data-testid="stSidebar"] {{
    background-color: {THEME['off_white']} !important;
    border-right: 3px solid {THEME['gold']} !important;
}}

section[data-testid="stSidebar"] .block-container {{
    padding: 1.5rem 1rem !important;
}}

.sidebar-header {{
    font-family: 'Crimson Text', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: {THEME['burgundy']};
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid {THEME['light_gold']};
}}

.sidebar-section {{
    margin-bottom: 1.8rem;
}}

.sidebar-section-title {{
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {THEME['gold']};
    margin-bottom: 0.8rem;
}}

.document-item {{
    background: {THEME['cream']};
    border: 1px solid {THEME['border']};
    border-radius: 6px;
    padding: 0.9rem;
    margin-bottom: 0.6rem;
    cursor: pointer;
    transition: all 0.2s;
}}

.document-item:hover {{
    border-color: {THEME['gold']};
    box-shadow: 0 2px 8px rgba(212, 175, 55, 0.15);
}}

.document-item-name {{
    font-family: 'Crimson Text', serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: {THEME['burgundy']};
    margin-bottom: 0.3rem;
}}

.document-item-meta {{
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    color: {THEME['gold']};
}}

/* ═════════════════════════════════════════ */
/* TOP BRAND BAR */
/* ═════════════════════════════════════════ */
.brand-bar {{
    background: linear-gradient(135deg, {THEME['burgundy']} 0%, {THEME['burgundy_light']} 100%);
    color: {THEME['off_white']};
    padding: 1.2rem 2rem;
    border-bottom: 3px solid {THEME['gold']};
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}}

.brand-left {{
    display: flex;
    align-items: center;
    gap: 1rem;
}}

.brand-title {{
    font-family: 'Crimson Text', serif;
    font-size: 1.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
}}

.brand-subtitle {{
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    color: {THEME['light_gold']};
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}}

.brand-right {{
    display: flex;
    align-items: center;
    gap: 1.5rem;
}}

.search-box {{
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid {THEME['light_gold']};
    border-radius: 6px;
    padding: 0.6rem 1rem;
    color: {THEME['off_white']};
    font-family: 'Crimson Text', serif;
    font-size: 0.9rem;
    width: 200px;
}}

.search-box:focus {{
    outline: none;
    box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.3);
}}

.theme-toggle {{
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid {THEME['light_gold']};
    color: {THEME['off_white']};
    padding: 0.6rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 1rem;
    transition: all 0.2s;
}}

.theme-toggle:hover {{
    background: rgba(212, 175, 55, 0.3);
}}

/* ═════════════════════════════════════════ */
/* MAIN WORKSPACE */
/* ═════════════════════════════════════════ */
.workspace-container {{
    display: flex;
    height: calc(100vh - 80px);
    gap: 2px;
    background: {THEME['shadow']};
}}

.pdf-pane {{
    flex: 0 0 48%;
    display: flex;
    flex-direction: column;
    background: {THEME['off_white']};
    border-right: 3px solid {THEME['gold']};
    overflow: hidden;
}}

.pdf-header {{
    background: linear-gradient(135deg, {THEME['burgundy_light']} 0%, {THEME['burgundy']} 100%);
    color: {THEME['off_white']};
    padding: 1.2rem 1.5rem;
    border-bottom: 1px solid {THEME['gold']};
}}

.pdf-header-title {{
    font-family: 'Crimson Text', serif;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}}

.pdf-header-meta {{
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: {THEME['light_gold']};
}}

.pdf-viewer {{
    flex: 1;
    overflow-y: auto;
    padding: 0;
}}

.pdf-viewer iframe {{
    width: 100%;
    height: 100%;
    border: none;
}}

/* ═════════════════════════════════════════ */
/* Q&A PANE */
/* ═════════════════════════════════════════ */
.qa-pane {{
    flex: 0 0 52%;
    display: flex;
    flex-direction: column;
    background: {THEME['cream']};
    overflow: hidden;
}}

.qa-header {{
    background: linear-gradient(135deg, {THEME['burgundy']} 0%, {THEME['burgundy_light']} 100%);
    color: {THEME['off_white']};
    padding: 1.5rem;
    border-bottom: 3px solid {THEME['gold']};
}}

.qa-header-title {{
    font-family: 'Crimson Text', serif;
    font-size: 1.3rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}}

.qa-header-meta {{
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: {THEME['light_gold']};
    opacity: 0.9;
}}

.qa-filters {{
    padding: 1rem 1.5rem;
    background: {THEME['off_white']};
    border-bottom: 1px solid {THEME['border']};
    display: flex;
    gap: 1rem;
    align-items: center;
}}

.filter-label {{
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    color: {THEME['gold']};
}}

.qa-messages {{
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
}}

.message {{
    animation: slideIn 0.3s ease;
}}

@keyframes slideIn {{
    from {{
        opacity: 0;
        transform: translateY(10px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

.message-user {{
    align-self: flex-end;
    max-width: 85%;
    background: {THEME['gold']};
    color: {THEME['burgundy']};
    padding: 1rem 1.2rem;
    border-radius: 8px 8px 2px 8px;
    box-shadow: 0 2px 6px rgba(212, 175, 55, 0.2);
}}

.message-user-text {{
    font-family: 'Crimson Text', serif;
    font-size: 1rem;
    line-height: 1.6;
}}

.message-ai {{
    align-self: flex-start;
    max-width: 85%;
    background: {THEME['off_white']};
    color: {THEME['burgundy']};
    padding: 1rem 1.2rem;
    border-radius: 8px 8px 8px 2px;
    border-left: 4px solid {THEME['gold']};
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}

.message-ai-text {{
    font-family: 'Crimson Text', serif;
    font-size: 0.98rem;
    line-height: 1.7;
}}

.message-time {{
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    color: #999;
    margin-top: 0.4rem;
}}

.message-evidence {{
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: {THEME['gold']};
    margin-top: 0.6rem;
    padding-top: 0.6rem;
    border-top: 1px solid {THEME['border']};
}}

/* ═════════════════════════════════════════ */
/* INPUT SECTION */
/* ═════════════════════════════════════════ */
.qa-input-section {{
    padding: 1.5rem;
    border-top: 2px solid {THEME['light_gold']};
    background: {THEME['off_white']};
}}

.suggested-prompts {{
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}}

.prompt-chip {{
    padding: 0.6rem 1rem;
    background: {THEME['cream']};
    color: {THEME['burgundy']};
    border: 1px solid {THEME['gold']};
    border-radius: 20px;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s;
}}

.prompt-chip:hover {{
    background: {THEME['gold']};
    color: {THEME['burgundy']};
}}

.qa-input-box {{
    display: flex;
    gap: 0.8rem;
}}

.qa-input {{
    flex: 1;
    padding: 0.9rem 1.2rem;
    border: 2px solid {THEME['light_gold']};
    border-radius: 6px;
    font-family: 'Crimson Text', serif;
    font-size: 0.95rem;
    color: {THEME['burgundy']};
    background: {THEME['cream']};
}}

.qa-input:focus {{
    outline: none;
    border-color: {THEME['gold']};
    box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.1);
}}

.qa-input::placeholder {{
    color: #999;
}}

.qa-send-btn {{
    padding: 0.9rem 1.5rem;
    background: {THEME['gold']};
    color: {THEME['burgundy']};
    border: none;
    border-radius: 6px;
    font-family: 'Crimson Text', serif;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 2px 6px rgba(212, 175, 55, 0.3);
}}

.qa-send-btn:hover {{
    background: {THEME['light_gold']};
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(212, 175, 55, 0.4);
}}

/* ═════════════════════════════════════════ */
/* LANDING PAGE */
/* ═════════════════════════════════════════ */
.landing-container {{
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, {THEME['cream']} 0%, {THEME['off_white']} 100%);
    padding: 2rem;
}}

.landing-box {{
    text-align: center;
    max-width: 600px;
}}

.landing-title {{
    font-family: 'Crimson Text', serif;
    font-size: 3rem;
    font-weight: 600;
    color: {THEME['burgundy']};
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}}

.landing-subtitle {{
    font-family: 'Crimson Text', serif;
    font-size: 1.3rem;
    color: {THEME['gold']};
    margin-bottom: 2rem;
    font-weight: 400;
}}

.landing-description {{
    font-size: 1rem;
    color: {THEME['burgundy']};
    margin-bottom: 2rem;
    line-height: 1.8;
}}

/* ═════════════════════════════════════════ */
/* SCROLLBAR */
/* ═════════════════════════════════════════ */
::-webkit-scrollbar {{
    width: 8px;
}}

::-webkit-scrollbar-track {{
    background: {THEME['cream']};
}}

::-webkit-scrollbar-thumb {{
    background: {THEME['light_gold']};
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: {THEME['gold']};
}}

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR WITH DOCUMENT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-header">
        📚 Your Documents
    </div>
    """, unsafe_allow_html=True)

    # Document Upload
    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-section-title">Upload New</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")

    # Document List
    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-section-title">Recent Documents</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.doc_name:
        st.markdown(f"""
        <div class="document-item">
            <div class="document-item-name">📄 {st.session_state.doc_name[:25]}...</div>
            <div class="document-item-meta">
                {st.session_state.doc_stats.get('pages', 0)} pages • {st.session_state.doc_stats.get('words', 0) // 1000}k words
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Settings Section
    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-section-title">Settings</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    with col2:
        if st.button("🔄 New Session"):
            st.session_state.answers = []
            st.session_state.chain = None
            st.session_state.doc_name = None
            st.rerun()

    # Info Section
    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-section-title">About</div>
        <p style="font-size: 0.8rem; color: {THEME['burgundy']}; margin-top: 0.8rem;">
            <strong>Scriptorium</strong> is an AI research assistant that helps you analyze documents 
            with precision using RAG technology.
        </p>
        <p style="font-size: 0.75rem; color: {THEME['gold']}; margin-top: 0.8rem;">
            Built with ❤️ by <strong>Ashlesha Sharma</strong>
        </p>
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
        template="""You are Scriptorium, a scholarly research assistant. Answer precisely from the context provided.

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

# ═════════════════════════════════════���═════════════════════════════════════════
# TOP BRAND BAR WITH SEARCH & THEME TOGGLE
# ═══════════════════════════════════════════════════════════════════════════════
col_brand_left, col_brand_right = st.columns([2, 1])

with col_brand_left:
    st.markdown(f"""
    <div class="brand-bar" style="border-right: none;">
        <div class="brand-left">
            <div>
                <div class="brand-title">📖 Scriptorium</div>
                <div class="brand-subtitle">Research Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.chain is None:
    st.markdown("""
    <div class="landing-container">
        <div class="landing-box">
            <div class="landing-title">Scriptorium</div>
            <div class="landing-subtitle">Research Intelligence for Any Subject</div>
            <div class="landing-description">
                Upload any research paper, thesis, or document and ask questions in natural language.
                Receive precise answers grounded in your document, with citations.
            </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if uploaded_file:
            try:
                pdf_bytes = uploaded_file.read()
                chain, stats = build_chain(pdf_bytes, uploaded_file.name)
                st.session_state.chain = chain
                st.session_state.doc_name = uploaded_file.name
                st.session_state.doc_stats = stats
                st.session_state.pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
                <div style="font-family: 'Crimson Text', serif; font-size: 1.1rem; color: {THEME['burgundy']}; margin-bottom: 1rem;">
                    Drop your PDF here or click to browse
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE - SPLIT SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
else:
    col_pdf, col_qa = st.columns([0.48, 0.52])

    # LEFT PANE: PDF
    with col_pdf:
        st.markdown(f"""
        <div class="pdf-pane" style="height: calc(100vh - 100px);">
            <div class="pdf-header">
                <div class="pdf-header-title">Document</div>
                <div class="pdf-header-meta">
                    Pages: {st.session_state.doc_stats["pages"]} • Words: {st.session_state.doc_stats["words"]//1000}k
                </div>
            </div>
            <div class="pdf-viewer">
                <iframe src="data:application/pdf;base64,{st.session_state.pdf_b64}" style="width:100%; height:100%; border:none;"></iframe>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # RIGHT PANE: Q&A
    with col_qa:
        st.markdown(f"""
        <div class="qa-pane" style="height: calc(100vh - 100px);">
            <div class="qa-header">
                <div class="qa-header-title">Ask Your Questions</div>
                <div class="qa-header-meta">{st.session_state.q_count} / {MAX_Q} questions used</div>
            </div>
            
            <div class="qa-filters">
                <div class="filter-label">🔍 Filter Results:</div>
        """, unsafe_allow_html=True)

        # Filter Section
        filter_cols = st.columns(2)
        with filter_cols[0]:
            relevance = st.slider("Relevance", 0.0, 1.0, 0.5, step=0.1)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Messages Display
        st.markdown(f"""
        <div class="qa-messages">
        """, unsafe_allow_html=True)

        for item in st.session_state.answers:
            if not item.get("error"):
                st.markdown(f"""
                <div class="message message-user">
                    <div class="message-user-text">{item['question']}</div>
                    <div class="message-time">{item.get('time', '')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="message message-ai">
                    <div class="message-ai-text">{item['content'][:500]}</div>
                    <div class="message-evidence">📌 Source: Page {item.get('sources', [{'page': '?'}])[0]['page']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Input Section
        if st.session_state.q_count < MAX_Q:
            st.markdown(f"""
            <div class="qa-input-section">
                <div class="suggested-prompts">
            """, unsafe_allow_html=True)

            prompts = [
                "📌 Summarize",
                "🔑 Key findings",
                "🔬 Methodology",
                "⚠️ Limitations",
                "📝 Conclusion"
            ]

            for prompt in prompts:
                if st.button(prompt, key=f"p_{prompt}"):
                    st.session_state.pending_q = prompt
                    st.session_state.thinking = True
                    st.session_state.q_count += 1
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            user_input = st.text_input("Ask me anything...", key="qa_input", placeholder="What would you like to know?")

            if user_input:
                st.session_state.pending_q = user_input
                st.session_state.thinking = True
                st.session_state.q_count += 1
                st.rerun()

            if st.session_state.thinking and st.session_state.pending_q:
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
                    st.session_state.thinking = False
                    st.session_state.pending_q = None
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
