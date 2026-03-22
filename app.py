"""
Scriptorium — AI research assistant for any subject.
Built by Ashlesha Sharma, Miranda House, Delhi University.

Alchemist's Archive design system.
Oxford Ink · Library Leather · Antique Brass · Parchment White.
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

OPENAI_API_KEY = get_api_key()
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

MAX_QUESTIONS = 10
MODEL         = "gpt-4o-mini"
EMBED_MODEL   = "text-embedding-3-small"
CHUNK_SIZE    = 900
CHUNK_OVERLAP = 180
TOP_K         = 4

st.set_page_config(
    page_title="Scriptorium",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400;1,500&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0D1117 !important;
    color: #E1D9CE !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 1.8rem 2rem 1.8rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #21262D !important;
    width: 220px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem !important;
}

/* ── Masthead ── */
.masthead {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 0.3rem;
}
.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: #B8860B;
    letter-spacing: 0.02em;
}
.masthead-tag {
    font-size: 0.65rem;
    color: #4A5568;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.masthead-sub {
    font-size: 0.72rem;
    color: #4A5568;
    font-weight: 300;
    margin-bottom: 1.4rem;
    letter-spacing: 0.03em;
}

/* ── Sidebar nav items ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.15s;
    font-size: 0.82rem;
    color: #6B7280;
    margin-bottom: 2px;
    letter-spacing: 0.03em;
}
.nav-item:hover { background: #161B22; color: #E1D9CE; }
.nav-item.active { background: #161B22; color: #B8860B; border-left: 2px solid #B8860B; }
.nav-icon { font-size: 0.9rem; opacity: 0.7; }

/* ── Usage meter ── */
.usage-wrap {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 4px;
    padding: 11px 14px;
    margin: 1rem 0;
}
.usage-top {
    display: flex;
    justify-content: space-between;
    margin-bottom: 7px;
    font-size: 0.7rem;
}
.usage-label { color: #4A5568; letter-spacing: 0.08em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; }
.usage-count { font-family: 'Playfair Display', serif; font-size: 0.9rem; color: #E1D9CE; }
.usage-track { height: 1px; background: #21262D; overflow: hidden; }
.usage-fill  { height: 100%; transition: width 0.5s ease; }
.usage-sub   { font-size: 0.65rem; color: #374151; margin-top: 5px; letter-spacing: 0.05em; font-family: 'JetBrains Mono', monospace; }

/* ── Sidebar how-it-works ── */
.how-step {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 0.75rem;
    line-height: 1.5;
    color: #4A5568;
}
.how-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #008080;
    min-width: 14px;
    padding-top: 1px;
}

/* ── Section label ── */
.sec-label {
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #374151;
    font-family: 'JetBrains Mono', monospace;
    margin: 0 0 0.6rem 0;
}

/* ── Hero bar ── */
.hero-bar {
    background: #161B22;
    border: 1px solid #21262D;
    border-top: 2px solid #B8860B;
    border-radius: 0 0 4px 4px;
    padding: 1.2rem 1.8rem 1.4rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
.hero-left {}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #B8860B;
    margin-bottom: 6px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 500;
    color: #E1D9CE;
    line-height: 1.1;
    letter-spacing: -0.01em;
    margin: 0 0 4px 0;
}
.hero-title span { color: #B8860B; }
.hero-sub {
    font-size: 0.85rem;
    color: #4A5568;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.hero-right {
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #374151;
    letter-spacing: 0.05em;
    line-height: 1.8;
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 8px; margin: 0.8rem 0; }
.stat-card {
    flex: 1;
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 3px;
    padding: 10px 12px;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #B8860B44; }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 500;
    color: #B8860B;
    line-height: 1;
}
.stat-label {
    font-size: 0.58rem;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 3px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── PDF viewer ── */
.pdf-topbar {
    background: #161B22;
    border: 1px solid #21262D;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 8px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #4A5568;
    letter-spacing: 0.06em;
}
.pdf-topbar-name { color: #B8860B; }
.pdf-wrap {
    border: 1px solid #21262D;
    border-radius: 0 0 4px 4px;
    overflow: hidden;
    background: #161B22;
}
.pdf-placeholder {
    height: 420px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
}

/* ── Ready pill ── */
.ready-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #2D4F3E22;
    border: 1px solid #2D4F3E;
    color: #4ADE80;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
}
.ready-dot {
    width: 5px; height: 5px;
    background: #4ADE80;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.3;} }

/* ── Q&A panel ── */
.qa-header {
    margin-bottom: 1rem;
}
.qa-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 500;
    color: #E1D9CE;
    margin: 0 0 2px 0;
}
.qa-remaining {
    font-size: 0.75rem;
    color: #4A5568;
    letter-spacing: 0.04em;
}
.qa-remaining strong { color: #E1D9CE; font-weight: 500; }

/* ── Suggested prompts ── */
.prompt-label {
    font-size: 0.62rem;
    color: #374151;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.8rem 0 0.4rem;
}

/* ── Answer card ── */
.answer-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-left: 3px solid #B8860B;
    border-radius: 0 4px 4px 0;
    padding: 20px 22px;
    margin: 12px 0;
    position: relative;
}
.answer-q {
    font-family: 'Playfair Display', serif;
    font-size: 0.92rem;
    font-style: italic;
    color: #6B7280;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #21262D;
}
.answer-q::before { content: '"'; }
.answer-q::after  { content: '"'; }
.answer-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #B8860B;
    margin: 14px 0 5px 0;
}
.answer-body {
    font-size: 0.88rem;
    color: #C9B99A;
    line-height: 1.8;
    font-weight: 300;
}
.answer-points {
    margin: 0;
    padding-left: 16px;
    font-size: 0.85rem;
    color: #C9B99A;
    line-height: 1.8;
    font-weight: 300;
}
.answer-points li { margin-bottom: 3px; }

/* ── Evidence block ── */
.evidence-block {
    background: #0D1117;
    border-left: 2px solid #7E2F2F;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 0 3px 3px 0;
}
.evidence-text {
    font-size: 0.8rem;
    color: #6B7280;
    font-style: italic;
    line-height: 1.6;
}
.evidence-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: #7E2F2F;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 5px;
    font-style: normal;
}
.answer-time {
    font-size: 0.6rem;
    color: #21262D;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 12px;
    text-align: right;
    letter-spacing: 0.05em;
}

/* ── Thinking state — amber candle pulse ── */
.thinking-wrap {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 4px;
    margin: 10px 0;
}
.candle-glow {
    width: 10px; height: 10px;
    background: #FFBF00;
    border-radius: 50%;
    box-shadow: 0 0 12px #FFBF0066, 0 0 4px #FFBF00;
    animation: flicker 1.2s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes flicker {
    0%,100%{ opacity:1; box-shadow:0 0 12px #FFBF0066,0 0 4px #FFBF00; }
    40%    { opacity:.7; box-shadow:0 0 6px #FFBF0033,0 0 2px #FFBF00; }
    70%    { opacity:.9; box-shadow:0 0 16px #FFBF0088,0 0 6px #FFBF00; }
}
.thinking-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #FFBF00;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.thinking-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #FFBF00 0%, transparent 100%);
    opacity: 0.3;
}

/* ── Empty states ── */
.empty-qa {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 4px;
    padding: 3.5rem 2rem;
    text-align: center;
}
.empty-icon { font-size: 1.8rem; opacity: 0.25; margin-bottom: 0.8rem; }
.empty-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: #374151;
    margin-bottom: 0.3rem;
    font-weight: 400;
}
.empty-sub { font-size: 0.78rem; color: #21262D; font-weight: 300; }

/* ── Limit wall ── */
.limit-wall {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 4px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: #0D1117 !important;
    border: 1px solid #21262D !important;
    color: #4A5568 !important;
    border-radius: 3px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    transition: all .2s !important;
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    padding: 7px 12px !important;
}
.stButton > button:hover {
    border-color: #B8860B55 !important;
    color: #B8860B !important;
    background: #161B22 !important;
}
div[data-testid="stChatInput"] textarea {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    color: #E1D9CE !important;
    border-radius: 3px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #B8860B !important;
    box-shadow: 0 0 0 2px #B8860B18 !important;
}
div[data-testid="stFileUploader"] {
    background: #161B22 !important;
    border: 1px dashed #21262D !important;
    border-radius: 3px !important;
}
.streamlit-expanderHeader {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 3px !important;
    color: #4A5568 !important;
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stProgress > div > div { background: #B8860B !important; }
hr { border-color: #21262D !important; opacity: 1 !important; }

/* ── Plotly dark override ── */
.js-plotly-plot .plotly { border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "answers":    [],
    "chain":      None,
    "doc_name":   None,
    "doc_stats":  None,
    "pdf_bytes":  None,
    "q_count":    0,
    "pending_q":  None,
    "thinking":   False,
    "last_q":     None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="masthead">
        <div class="masthead-title">📜 Scriptorium</div>
    </div>
    <div class="masthead-sub">Any paper. Any subject. Instant insight.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="nav-item active">
        <span class="nav-icon">✦</span> New Session
    </div>
    <div class="nav-item">
        <span class="nav-icon">◷</span> History
    </div>
    <div class="nav-item">
        <span class="nav-icon">⊟</span> Saved Papers
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="margin:1rem 0">', unsafe_allow_html=True)

    used  = st.session_state.q_count
    left  = MAX_QUESTIONS - used
    pct   = int((used / MAX_QUESTIONS) * 100)
    bar_c = "#B8860B" if pct < 60 else "#C4845A" if pct < 90 else "#7E2F2F"

    st.markdown(f"""
    <div class="usage-wrap">
        <div class="usage-top">
            <span class="usage-label">Session</span>
            <span class="usage-count">{used} / {MAX_QUESTIONS}</span>
        </div>
        <div class="usage-track">
            <div class="usage-fill" style="width:{pct}%;background:{bar_c}"></div>
        </div>
        <div class="usage-sub">{left} question{'s' if left!=1 else ''} remaining · refresh to reset</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label" style="margin-top:1.2rem">How it works</div>', unsafe_allow_html=True)

    steps = [
        ("i", "PDF split into overlapping passages"),
        ("ii", "Passages → embedding vectors"),
        ("iii", "Question → nearest vectors found"),
        ("iv", "Passages + question → GPT"),
        ("v", "Answer from your document only"),
    ]
    html_steps = "".join(f"""
    <div class="how-step">
        <span class="how-num">{n}.</span>
        <span>{t}</span>
    </div>""" for n, t in steps)
    st.markdown(html_steps, unsafe_allow_html=True)

    st.markdown('<hr style="margin:1rem 0">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.68rem;color:#374151;line-height:1.9;font-weight:300;font-family:'JetBrains Mono',monospace">
        Built by<br>
        <span style="color:#B8860B">Ashlesha Sharma</span><br>
        BSc (H) Chemistry<br>
        Miranda House · DU<br><br>
        <a href="https://github.com/ashlesha-sharma/cheminsight-ai"
           style="color:#008080;text-decoration:none;font-size:.65rem">
           github ↗
        </a>
    </div>
    """, unsafe_allow_html=True)


# ── API key guard ──────────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    st.error("No API key found. Add OPENAI_API_KEY to Streamlit secrets.")
    st.stop()


# ── Core functions ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_chain(_pdf_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_pdf_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages  = loader.load()
    os.unlink(tmp_path)

    total_text = " ".join(p.page_content for p in pages).strip()
    if len(total_text) < 150:
        raise ValueError("SCANNED_PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    embeddings  = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=f"./chroma_{filename[:14].replace('.','_')}",
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3},
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are Scriptorium — a precise, scholarly research assistant trained to extract and explain information from academic papers.

You assist researchers across all disciplines: science, humanities, social science, law, medicine, economics, and more.

Rules:
- Answer ONLY from the context below. If absent: state "This is not covered in the uploaded paper."
- Quote exact values, statistics, dates, names where present.
- Structure your answer in three parts:
  1. Direct answer (1–3 sentences, precise)
  2. Key points (3–5 bullets if applicable)
  3. One direct quote or evidence from the text

Context:
──────────────────────
{context}
──────────────────────

Question: {question}

Answer:""",
    )

    llm   = ChatOpenAI(model_name=MODEL, temperature=0.1)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    stats = {
        "pages":        len(pages),
        "chunks":       len(chunks),
        "words":        len(total_text.split()),
        "chars":        len(total_text),
        "read_min":     max(1, len(total_text.split()) // 250),
        "page_lengths": [len(p.page_content) for p in pages],
    }
    return chain, stats


def get_sources(source_docs):
    out, seen = [], set()
    for doc in source_docs:
        page = doc.metadata.get("page", 0)
        if page not in seen:
            seen.add(page)
            snippet = doc.page_content[:220].replace("\n", " ").strip()
            out.append({"page": page + 1, "snippet": snippet})
    return out


def parse_answer(raw):
    lines = raw.strip().split("\n")
    direct, points, evidence = [], [], []
    mode = "direct"
    for line in lines:
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if any(k in low for k in ["key point","key finding","important","notable point"]):
            mode = "points"; continue
        if any(k in low for k in ["evidence","quote","from the paper","the paper state","according to"]):
            mode = "evidence"; continue
        if s.startswith(("•", "-", "*", "–")):
            points.append(s.lstrip("•-*– ")); mode = "points"
        elif s.startswith(('"', "'")):
            evidence.append(s.strip("\"'"))
        else:
            if mode == "direct": direct.append(s)
            elif mode == "points": points.append(s)
            elif mode == "evidence": evidence.append(s)
    return {
        "direct":   " ".join(direct) if direct else raw.strip(),
        "points":   points[:6],
        "evidence": evidence[:1],
    }


def density_chart(page_lengths):
    avg    = sum(page_lengths) / len(page_lengths)
    colors = ["#B8860B" if l >= avg else "#21262D" for l in page_lengths]
    fig    = go.Figure(go.Bar(
        x=list(range(1, len(page_lengths) + 1)),
        y=page_lengths,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="Page %{x} · %{y} chars<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Text density per page",
                   font=dict(size=10, color="#374151", family="JetBrains Mono"), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#374151", size=9),
        margin=dict(l=10, r=10, t=32, b=24),
        height=150,
        xaxis=dict(showgrid=False, zeroline=False,
                   title="pg", title_font_size=9,
                   tickfont_size=8, color="#374151",
                   linecolor="#21262D"),
        yaxis=dict(showgrid=True, gridcolor="#161B22",
                   zeroline=False, tickfont_size=8, color="#374151"),
        bargap=0.3,
    )
    return fig


# ── Hero bar ───────────────────────────────────────────────────────────────────
doc_name_display = st.session_state.doc_name or "No document loaded"
pages_display    = str(st.session_state.doc_stats["pages"]) + " pages" if st.session_state.doc_stats else "—"
words_display    = f"{st.session_state.doc_stats['words']//1000}k words" if st.session_state.doc_stats else "—"

st.markdown(f"""
<div class="hero-bar">
    <div class="hero-left">
        <div class="hero-eyebrow">📜 Research Intelligence · RAG · LangChain · OpenAI</div>
        <div class="hero-title">Scriptori<span>um</span></div>
        <div class="hero-sub">Upload any paper. Ask questions. Every answer from your document.</div>
    </div>
    <div class="hero-right">
        <div style="color:#B8860B;margin-bottom:4px">
            {doc_name_display[:42] + '…' if len(doc_name_display) > 42 else doc_name_display}
        </div>
        <div>{pages_display} &nbsp;·&nbsp; {words_display}</div>
        <div style="color:#21262D;margin-top:2px">Free to use · No account needed</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Main layout ────────────────────────────────────────────────────────────────
col_doc, col_qa = st.columns([1, 1.1], gap="large")


# ══ LEFT — Document panel ══════════════════════════════════════════════════════
with col_doc:

    st.markdown('<div class="sec-label">Document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Any PDF with selectable text — research papers, reports, textbooks.",
    )

    if uploaded:
        if st.session_state.doc_name != uploaded.name:
            st.session_state.answers  = []
            st.session_state.q_count  = 0
            st.session_state.thinking = False
            st.session_state.last_q   = None

            bar = st.progress(0, text="Opening document...")
            time.sleep(0.15)
            bar.progress(20, text="Splitting into passages...")
            time.sleep(0.1)
            bar.progress(45, text="Building vector index — ~20s...")

            try:
                pdf_bytes = uploaded.read()
                chain, stats = build_chain(pdf_bytes, uploaded.name)
                bar.progress(90, text="Finalising...")
                time.sleep(0.3)
                bar.progress(100, text="Ready.")
                time.sleep(0.4)
                bar.empty()

                st.session_state.chain     = chain
                st.session_state.doc_name  = uploaded.name
                st.session_state.doc_stats = stats
                st.session_state.pdf_bytes = pdf_bytes

            except ValueError as e:
                bar.empty()
                if "SCANNED_PDF" in str(e):
                    st.error("Scanned PDF detected — no selectable text found. Try a PDF where text can be highlighted.")
                else:
                    st.error(f"Error: {e}")
            except Exception as e:
                bar.empty()
                err = str(e).lower()
                if "api" in err or "key" in err or "auth" in err:
                    st.error("API key error. Verify your key in Streamlit secrets.")
                elif "rate" in err:
                    st.error("Rate limit reached. Wait 60 seconds and try again.")
                else:
                    st.error(f"Unexpected error: {e}")

    # ── Stats + PDF viewer ─────────────────────────────────────────────────────
    if st.session_state.doc_stats:
        s  = st.session_state.doc_stats
        dn = st.session_state.doc_name

        st.markdown(f"""
        <div class="ready-pill">
            <span class="ready-dot"></span>
            Indexed · ready
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-num">{s['pages']}</div>
                <div class="stat-label">Pages</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{s['chunks']}</div>
                <div class="stat-label">Passages</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{s['read_min']}m</div>
                <div class="stat-label">Read time</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{s['words']//1000}k</div>
                <div class="stat-label">Words</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if len(s["page_lengths"]) > 1:
            st.plotly_chart(
                density_chart(s["page_lengths"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # ── PDF viewer ─────────────────────────────────────────────────────────
        short_name = dn[:38] + "…" if len(dn) > 38 else dn
        st.markdown(f"""
        <div class="pdf-topbar">
            <span>FileName: <span class="pdf-topbar-name">{short_name}</span></span>
            <span>[Pages: {s['pages']}]</span>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.pdf_bytes:
            b64 = base64.b64encode(st.session_state.pdf_bytes).decode("utf-8")
            st.markdown(f"""
            <div class="pdf-wrap">
                <iframe
                    src="data:application/pdf;base64,{b64}"
                    width="100%"
                    height="420px"
                    style="border:none;display:block;background:#161B22"
                ></iframe>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="pdf-topbar">
            <span>FileName: <span style="color:#374151">no document loaded</span></span>
            <span>[Pages: —]</span>
        </div>
        <div class="pdf-wrap">
            <div class="pdf-placeholder">
                <div style="font-size:2.5rem;opacity:.12">📄</div>
                <div style="font-family:'Playfair Display',serif;font-size:.95rem;
                            color:#21262D;font-weight:400">
                    Drop a document above
                </div>
                <div style="font-size:.72rem;color:#1C2128;font-weight:300;
                            font-family:'JetBrains Mono',monospace">
                    Research paper · Thesis · Report · Textbook chapter
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══ RIGHT — Q&A panel ══════════════════════════════════════════════════════════
with col_qa:

    remaining = MAX_QUESTIONS - st.session_state.q_count
    st.markdown(f"""
    <div class="qa-header">
        <div class="qa-title">Ask Your Questions</div>
        <div class="qa-remaining">
            <strong>{remaining}</strong> / {MAX_QUESTIONS} remaining
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.chain is None:
        st.markdown("""
        <div class="empty-qa">
            <div class="empty-icon">✦</div>
            <div class="empty-title">Upload a paper to begin</div>
            <div class="empty-sub">Ask about findings, methods, arguments, data — anything in the document</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Input + suggested prompts ──────────────────────────────────────────
        if st.session_state.q_count < MAX_QUESTIONS:
            user_input = st.chat_input(
                "Ask a question about the paper..."
            )

            st.markdown('<div class="prompt-label">Suggested Prompts</div>', unsafe_allow_html=True)

            prompts = [
                "Summarise this paper",
                "What are the key findings?",
                "What methodology is used?",
                "What limitations are mentioned?",
                "What does the paper conclude?",
                "What evidence supports the main argument?",
            ]
            for p in prompts:
                if st.button(f"· {p}", key=f"p_{p}"):
                    st.session_state.pending_q = p
                    st.rerun()

        # ── Answer cards ───────────────────────────────────────────────────────
        for item in st.session_state.answers:
            if item.get("error"):
                st.error(item["content"])
                continue

            parsed = parse_answer(item["content"])

            pts_html = ""
            if parsed["points"]:
                lis = "".join(f"<li>{pt}</li>" for pt in parsed["points"])
                pts_html = f"""
                <div class="answer-section">Key Points</div>
                <ul class="answer-points">{lis}</ul>"""

            ev_html = ""
            if item.get("sources"):
                src = item["sources"][0]
                ev_html = f"""
                <div class="answer-section">Evidence from Paper</div>
                <div class="evidence-block">
                    <div class="evidence-text">"{src['snippet']}…"</div>
                    <div class="evidence-tag">◼ Archival Ref · Page {src['page']}</div>
                </div>"""

            st.markdown(f"""
            <div class="answer-card">
                <div class="answer-q">{item['question']}</div>
                <div class="answer-section">Answer</div>
                <div class="answer-body">{parsed['direct']}</div>
                {pts_html}
                {ev_html}
                <div class="answer-time">{item.get('time','')}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Thinking state ─────────────────────────────────────────────────────
        if st.session_state.thinking:
            st.markdown("""
            <div class="thinking-wrap">
                <div class="candle-glow"></div>
                <div class="thinking-text">Analysing document</div>
                <div class="thinking-line"></div>
            </div>
            """, unsafe_allow_html=True)

        # ── Process question ───────────────────────────────────────────────────
        question = st.session_state.pending_q or (user_input if st.session_state.chain else None)

        if question and not st.session_state.thinking and st.session_state.q_count < MAX_QUESTIONS:
            st.session_state.pending_q = None
            st.session_state.last_q    = question
            st.session_state.thinking  = True
            st.session_state.q_count  += 1
            st.rerun()

        if st.session_state.thinking and st.session_state.last_q:
            q = st.session_state.last_q
            try:
                result  = st.session_state.chain.invoke({"question": q})
                answer  = result["answer"]
                sources = get_sources(result.get("source_documents", []))
                st.session_state.answers.append({
                    "question":  q,
                    "content":   answer,
                    "sources":   sources,
                    "time":      datetime.datetime.now().strftime("%H:%M"),
                    "error":     False,
                })
            except Exception as e:
                err = str(e).lower()
                if "rate" in err:
                    msg = "Rate limit reached — wait 60 seconds."
                elif "api" in err or "key" in err:
                    msg = "API key error — verify your Streamlit secrets."
                else:
                    msg = f"Error: {e}"
                st.session_state.answers.append({
                    "error": True, "content": msg,
                })
            finally:
                st.session_state.thinking = False
                st.session_state.last_q   = None
                st.rerun()

        # ── Session limit ──────────────────────────────────────────────────────
        if st.session_state.q_count >= MAX_QUESTIONS:
            st.markdown("""
            <div class="limit-wall">
                <div style="font-family:'Playfair Display',serif;font-size:1.05rem;
                            color:#6B7280;margin-bottom:.4rem;font-weight:400">
                    Session limit reached
                </div>
                <div style="font-size:.78rem;color:#374151;font-weight:300;
                            font-family:'JetBrains Mono',monospace">
                    10 / 10 questions used · refresh the page to begin a new session
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Clear ──────────────────────────────────────────────────────────────
        if st.session_state.answers:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("Clear answers", key="clear_btn"):
                st.session_state.answers  = []
                st.session_state.thinking = False
                st.session_state.last_q   = None
                if st.session_state.chain:
                    st.session_state.chain.memory.clear()
                st.rerun()
