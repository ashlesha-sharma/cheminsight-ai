"""
Scriptorium — research intelligence for any subject.
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

MAX_Q         = 10
MODEL         = "gpt-4o-mini"
EMBED_MODEL   = "text-embedding-3-small"
CHUNK_SIZE    = 900
CHUNK_OVERLAP = 180
TOP_K         = 4

st.set_page_config(
    page_title="Scriptorium",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;1,400&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400&display=swap');

/* ── palette ────────────────────────────────────────────
   One dark theme. Narrow tonal range. Accent used once.
   bg0   #0C0F14   deepest base
   bg1   #131720   panels
   bg2   #181D28   elevated cards
   bd0   #1A2030   faint border
   bd1   #222B3A   default border
   t0    #C4BBa8   primary text   (warm off-white)
   t1    #4E5768   secondary text
   t2    #272F3E   muted / disabled
   acc   #8A6E3C   accent — muted gold, used sparingly
──────────────────────────────────────────────────────── */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0C0F14 !important;
    color: #C4BBA8 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] {
    background: #0C0F14 !important;
    border-right: 1px solid #1A2030 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.4rem 1rem !important;
}

/* sidebar brand */
.sb-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 500;
    color: #C4BBA8;
    letter-spacing: 0.02em;
    margin-bottom: 1.6rem;
}
.sb-section {
    font-size: 0.6rem;
    font-weight: 400;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #272F3E;
    font-family: 'JetBrains Mono', monospace;
    margin: 0 0 0.5rem 0;
}
.sb-new-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 8px 10px;
    background: #131720;
    border: 1px solid #1A2030;
    border-radius: 4px;
    color: #4E5768;
    font-size: 0.78rem;
    cursor: pointer;
    transition: all 0.15s;
    margin-bottom: 1rem;
    font-family: 'Inter', sans-serif;
}
.sb-new-btn:hover { border-color: #222B3A; color: #C4BBA8; }
.sb-file-item {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 9px 10px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.12s;
    margin-bottom: 1px;
}
.sb-file-item:hover  { background: #131720; }
.sb-file-item.active { background: #131720; border-left: 2px solid #8A6E3C; padding-left: 8px; }
.sb-file-icon {
    width: 28px; height: 28px;
    background: #181D28;
    border: 1px solid #1A2030;
    border-radius: 3px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;
    color: #4E5768; flex-shrink: 0;
}
.sb-file-name {
    font-size: 0.78rem;
    color: #C4BBA8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 140px;
}
.sb-file-meta {
    font-size: 0.65rem;
    color: #272F3E;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 1px;
}
.sb-usage {
    background: #131720;
    border: 1px solid #1A2030;
    border-radius: 4px;
    padding: 10px 12px;
    margin-top: 1rem;
}
.sb-usage-row {
    display: flex; justify-content: space-between;
    font-size: 0.68rem; margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.sb-usage-label { color: #272F3E; }
.sb-usage-count { color: #4E5768; }
.sb-track { height: 1px; background: #1A2030; border-radius: 1px; overflow: hidden; }
.sb-fill  { height: 100%; background: #8A6E3C; transition: width 0.4s; }
.sb-about {
    font-size: 0.68rem;
    color: #272F3E;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.8;
    margin-top: 1.4rem;
}
.sb-about a { color: #4E5768; text-decoration: none; }

/* ── landing (before upload) ─────────────────────────── */
.landing-wrap {
    min-height: 85vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 2rem;
}
.landing-greeting {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 400;
    color: #C4BBA8;
    text-align: center;
    margin-bottom: 0.5rem;
    letter-spacing: -0.01em;
}
.landing-sub {
    font-size: 0.9rem;
    color: #4E5768;
    text-align: center;
    font-weight: 300;
    margin-bottom: 2.5rem;
    letter-spacing: 0.02em;
}
.upload-card {
    width: 100%;
    max-width: 520px;
    background: #131720;
    border: 1px solid #1A2030;
    border-radius: 6px;
    padding: 2.5rem 2rem;
    text-align: center;
}
.upload-card-title {
    font-size: 0.85rem;
    color: #C4BBA8;
    font-weight: 400;
    margin-bottom: 0.3rem;
}
.upload-card-sub {
    font-size: 0.75rem;
    color: #4E5768;
    font-weight: 300;
    margin-bottom: 1.4rem;
}
.landing-note {
    font-size: 0.72rem;
    color: #272F3E;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 1.6rem;
    letter-spacing: 0.04em;
    text-align: center;
}

/* ── workspace (after upload) ────────────────────────── */
.workspace { padding: 1.4rem 1.8rem 2rem; }
.pane-label {
    font-size: 0.6rem;
    font-weight: 400;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #272F3E;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.6rem;
}
.pdf-bar {
    background: #131720;
    border: 1px solid #1A2030;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 7px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #4E5768;
}
.pdf-bar-name { color: #8A6E3C; }
.pdf-frame-wrap {
    border: 1px solid #1A2030;
    border-radius: 0 0 4px 4px;
    overflow: hidden;
}
.stat-row { display: flex; gap: 6px; margin: 0.7rem 0 1rem; }
.stat-card {
    flex: 1;
    background: #131720;
    border: 1px solid #1A2030;
    border-radius: 3px;
    padding: 9px 10px;
    text-align: center;
}
.stat-n {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #C4BBA8;
    line-height: 1;
}
.stat-l {
    font-size: 0.58rem;
    color: #272F3E;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Q&A panel ───────────────────────────────────────── */
.qa-title-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 1rem;
}
.qa-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 400;
    color: #C4BBA8;
}
.qa-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #4E5768;
}
.qa-count strong { color: #C4BBA8; font-weight: 400; }

/* prompt chips */
.prompt-chips { display: flex; flex-direction: column; gap: 4px; margin-bottom: 1rem; }
.prompt-chip {
    padding: 7px 12px;
    background: #131720;
    border: 1px solid #1A2030;
    border-radius: 3px;
    font-size: 0.78rem;
    color: #4E5768;
    cursor: pointer;
    transition: all 0.12s;
}
.prompt-chip:hover { border-color: #222B3A; color: #C4BBA8; }

/* answer card */
.a-card {
    background: #131720;
    border: 1px solid #1A2030;
    border-left: 2px solid #8A6E3C;
    border-radius: 0 4px 4px 0;
    padding: 18px 20px;
    margin: 10px 0;
}
.a-q {
    font-family: 'Playfair Display', serif;
    font-size: 0.88rem;
    font-style: italic;
    color: #4E5768;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1A2030;
}
.a-sec {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.56rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #8A6E3C;
    margin: 12px 0 4px;
}
.a-body {
    font-size: 0.85rem;
    color: #B0A896;
    line-height: 1.8;
    font-weight: 300;
}
.a-pts {
    margin: 0; padding-left: 14px;
    font-size: 0.82rem;
    color: #B0A896;
    line-height: 1.8;
    font-weight: 300;
}
.a-pts li { margin-bottom: 2px; }
.evidence {
    background: #0C0F14;
    border-left: 1px solid #4E5768;
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 0 2px 2px 0;
}
.evidence-text {
    font-size: 0.78rem;
    color: #4E5768;
    font-style: italic;
    line-height: 1.55;
}
.evidence-ref {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.56rem;
    color: #272F3E;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.a-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: #1A2030;
    text-align: right;
    margin-top: 10px;
}

/* thinking */
.thinking {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px;
    background: #131720;
    border: 1px solid #1A2030;
    border-radius: 4px;
    margin: 8px 0;
}
.t-dot {
    width: 7px; height: 7px;
    background: #8A6E3C;
    border-radius: 50%;
    animation: tp 1.4s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes tp { 0%,100%{opacity:.9;} 50%{opacity:.25;} }
.t-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #4E5768;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* empty / limit */
.empty-qa {
    background: #131720;
    border: 1px solid #1A2030;
    border-radius: 4px;
    padding: 3rem 2rem;
    text-align: center;
    margin-top: 0.5rem;
}
.empty-qa-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    color: #4E5768;
    font-weight: 400;
    margin-bottom: 0.3rem;
}
.empty-qa-sub { font-size: 0.75rem; color: #272F3E; font-weight: 300; }
.limit-wall {
    background: #131720;
    border: 1px solid #1A2030;
    border-radius: 4px;
    padding: 1.6rem;
    text-align: center;
    margin-top: 1rem;
}

/* streamlit overrides */
.stButton > button {
    background: #131720 !important;
    border: 1px solid #1A2030 !important;
    color: #4E5768 !important;
    border-radius: 3px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    transition: all .15s !important;
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    padding: 7px 12px !important;
}
.stButton > button:hover {
    border-color: #222B3A !important;
    color: #C4BBA8 !important;
    background: #181D28 !important;
}
div[data-testid="stChatInput"] textarea {
    background: #131720 !important;
    border: 1px solid #1A2030 !important;
    color: #C4BBA8 !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #8A6E3C !important;
    box-shadow: none !important;
}
div[data-testid="stFileUploader"] {
    background: #131720 !important;
    border: 1px dashed #1A2030 !important;
    border-radius: 4px !important;
}
.stProgress > div > div { background: #8A6E3C !important; }
.streamlit-expanderHeader {
    background: #131720 !important;
    border: 1px solid #1A2030 !important;
    border-radius: 3px !important;
    color: #4E5768 !important;
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.04em !important;
}
hr { border-color: #1A2030 !important; opacity: 1 !important; }
</style>
""", unsafe_allow_html=True)


# ── session state ──────────────────────────────────────────────────────────────
for k, v in {
    "answers":   [],
    "chain":     None,
    "doc_name":  None,
    "doc_stats": None,
    "pdf_b64":   None,
    "q_count":   0,
    "pending_q": None,
    "thinking":  False,
    "last_q":    None,
    "sessions":  [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">Scriptorium</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Papers</div>', unsafe_allow_html=True)

    if st.button("+ New session", key="new_session"):
        for k in ["answers","chain","doc_name","doc_stats","pdf_b64","q_count","pending_q","thinking","last_q"]:
            st.session_state[k] = [] if k in ["answers","sessions"] else None if k not in ["q_count"] else 0
        st.session_state.thinking = False
        st.rerun()

    if st.session_state.doc_name:
        ext = st.session_state.doc_name.split(".")[-1].upper()[:3]
        short = st.session_state.doc_name[:22] + "…" if len(st.session_state.doc_name) > 22 else st.session_state.doc_name
        pages = st.session_state.doc_stats["pages"] if st.session_state.doc_stats else "—"
        st.markdown(f"""
        <div class="sb-file-item active">
            <div class="sb-file-icon">{ext}</div>
            <div>
                <div class="sb-file-name">{short}</div>
                <div class="sb-file-meta">{pages} pages</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:.72rem;color:#272F3E;font-family:'JetBrains Mono',monospace;
                    padding:8px 10px;letter-spacing:.04em">
            No papers loaded
        </div>
        """, unsafe_allow_html=True)

    used = st.session_state.q_count
    pct  = int((used / MAX_Q) * 100)
    st.markdown(f"""
    <div class="sb-usage">
        <div class="sb-usage-row">
            <span class="sb-usage-label">Session</span>
            <span class="sb-usage-count">{used} / {MAX_Q}</span>
        </div>
        <div class="sb-track">
            <div class="sb-fill" style="width:{pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-about">
        Ashlesha Sharma<br>
        Miranda House · DU<br>
        <a href="https://github.com/ashlesha-sharma/cheminsight-ai">github</a>
    </div>
    """, unsafe_allow_html=True)


# ── API key guard ──────────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    st.error("No API key found. Add OPENAI_API_KEY to Streamlit secrets.")
    st.stop()


# ── core functions ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_chain(_pdf_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_pdf_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages  = loader.load()
    os.unlink(tmp_path)

    total = " ".join(p.page_content for p in pages).strip()
    if len(total) < 150:
        raise ValueError("SCANNED_PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    embeddings  = OpenAIEmbeddings(model=EMBED_MODEL)
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
        template="""You are Scriptorium — a precise, scholarly research assistant for any academic discipline.

Rules:
- Answer ONLY from the context. If absent: state clearly it is not in the paper.
- Quote exact values, statistics, names where present.
- Structure: (1) direct answer, (2) key points if applicable, (3) evidence quote.

Context:
──────────────────────
{context}
──────────────────────

Question: {question}

Answer:""",
    )
    llm   = ChatOpenAI(model_name=MODEL, temperature=0.1)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    stats = {
        "pages":        len(pages),
        "chunks":       len(chunks),
        "words":        len(total.split()),
        "read_min":     max(1, len(total.split()) // 250),
        "page_lengths": [len(p.page_content) for p in pages],
    }
    return chain, stats


def get_sources(source_docs):
    out, seen = [], set()
    for doc in source_docs:
        page = doc.metadata.get("page", 0)
        if page not in seen:
            seen.add(page)
            snippet = doc.page_content[:200].replace("\n", " ").strip()
            out.append({"page": page + 1, "snippet": snippet})
    return out


def parse_answer(raw):
    lines = raw.strip().split("\n")
    direct, points, evidence = [], [], []
    mode = "direct"
    for line in lines:
        s = line.strip()
        if not s: continue
        low = s.lower()
        if any(k in low for k in ["key point","key finding","important point","notable"]):
            mode = "points"; continue
        if any(k in low for k in ["evidence","quote","from the paper","the paper state"]):
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
    colors = ["#8A6E3C" if l >= avg else "#1A2030" for l in page_lengths]
    fig    = go.Figure(go.Bar(
        x=list(range(1, len(page_lengths) + 1)),
        y=page_lengths,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="pg %{x} · %{y}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#272F3E", size=9),
        margin=dict(l=6, r=6, t=10, b=20),
        height=110,
        xaxis=dict(showgrid=False, zeroline=False,
                   title="", tickfont_size=8, color="#272F3E",
                   linecolor="#1A2030"),
        yaxis=dict(showgrid=True, gridcolor="#131720",
                   zeroline=False, tickfont_size=8, color="#272F3E",
                   showticklabels=False),
        bargap=0.3,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LANDING STATE — before upload
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.chain is None:

    st.markdown('<div class="landing-wrap">', unsafe_allow_html=True)

    now_h = datetime.datetime.now().hour
    greeting = "Good morning" if now_h < 12 else "Good afternoon" if now_h < 17 else "Good evening"

    st.markdown(f"""
    <div class="landing-greeting">{greeting}.</div>
    <div class="landing-sub">What would you like to research today?</div>
    """, unsafe_allow_html=True)

    col_c = st.columns([1, 2, 1])[1]
    with col_c:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="upload-card-title">Upload a paper to begin</div>
        <div class="upload-card-sub">
            Research paper &nbsp;·&nbsp; Thesis chapter &nbsp;·&nbsp; Report &nbsp;·&nbsp; Any PDF
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="landing-note">
            Chemistry &nbsp;·&nbsp; Economics &nbsp;·&nbsp; Law &nbsp;·&nbsp; Medicine &nbsp;·&nbsp; History &nbsp;·&nbsp; Any discipline
        </div>
        """, unsafe_allow_html=True)

    if uploaded:
        bar = st.progress(0, text="Reading document...")
        time.sleep(0.15)
        bar.progress(30, text="Splitting into passages...")
        time.sleep(0.1)
        bar.progress(55, text="Building index — ~20s...")
        try:
            pdf_bytes = uploaded.read()
            chain, stats = build_chain(pdf_bytes, uploaded.name)
            bar.progress(95, text="Finalising...")
            time.sleep(0.3)
            bar.progress(100, text="Ready.")
            time.sleep(0.4)
            bar.empty()
            st.session_state.chain    = chain
            st.session_state.doc_name = uploaded.name
            st.session_state.doc_stats = stats
            st.session_state.pdf_b64  = base64.b64encode(pdf_bytes).decode("utf-8")
            st.rerun()
        except ValueError as e:
            bar.empty()
            if "SCANNED_PDF" in str(e):
                st.error("Scanned PDF — no selectable text found. Try a PDF where text can be highlighted.")
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

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# WORKSPACE STATE — after upload
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="workspace">', unsafe_allow_html=True)

    doc_col, qa_col = st.columns([1, 1.1], gap="large")

    # ── LEFT: document ─────────────────────────────────────────────────────────
    with doc_col:
        s  = st.session_state.doc_stats
        dn = st.session_state.doc_name

        st.markdown('<div class="pane-label">Document</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-n">{s['pages']}</div>
                <div class="stat-l">Pages</div>
            </div>
            <div class="stat-card">
                <div class="stat-n">{s['chunks']}</div>
                <div class="stat-l">Passages</div>
            </div>
            <div class="stat-card">
                <div class="stat-n">{s['read_min']}m</div>
                <div class="stat-l">Read time</div>
            </div>
            <div class="stat-card">
                <div class="stat-n">{s['words']//1000}k</div>
                <div class="stat-l">Words</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if len(s["page_lengths"]) > 1:
            st.plotly_chart(
                density_chart(s["page_lengths"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        short = dn[:36] + "…" if len(dn) > 36 else dn
        st.markdown(f"""
        <div class="pdf-bar">
            <span class="pdf-bar-name">{short}</span>
            <span>{s['pages']} pages</span>
        </div>
        <div class="pdf-frame-wrap">
            <iframe
                src="data:application/pdf;base64,{st.session_state.pdf_b64}"
                width="100%" height="440px"
                style="border:none;display:block;background:#0C0F14"
            ></iframe>
        </div>
        """, unsafe_allow_html=True)

    # ── RIGHT: Q&A ─────────────────────────────────────────────────────────────
    with qa_col:
        remaining = MAX_Q - st.session_state.q_count
        st.markdown(f"""
        <div class="qa-title-row">
            <div class="qa-title">Ask your questions</div>
            <div class="qa-count"><strong>{remaining}</strong> / {MAX_Q} remaining</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.q_count < MAX_Q:
            user_input = st.chat_input("Ask anything about the paper...")

            st.markdown('<div class="pane-label" style="margin-top:.6rem">Suggested</div>', unsafe_allow_html=True)
            for p in [
                "Summarise this paper",
                "What are the key findings?",
                "What methodology is used?",
                "What limitations are mentioned?",
                "What does the paper conclude?",
            ]:
                if st.button(p, key=f"p_{p}"):
                    st.session_state.pending_q = p
                    st.rerun()

        if not st.session_state.answers:
            st.markdown("""
            <div class="empty-qa">
                <div class="empty-qa-title">Ready when you are</div>
                <div class="empty-qa-sub">Ask about findings, methods, arguments, data</div>
            </div>
            """, unsafe_allow_html=True)

        for item in st.session_state.answers:
            if item.get("error"):
                st.error(item["content"])
                continue
            parsed = parse_answer(item["content"])
            pts_html = ""
            if parsed["points"]:
                lis = "".join(f"<li>{pt}</li>" for pt in parsed["points"])
                pts_html = f'<div class="a-sec">Key points</div><ul class="a-pts">{lis}</ul>'
            ev_html = ""
            if item.get("sources"):
                src = item["sources"][0]
                ev_html = f"""
                <div class="a-sec">Evidence</div>
                <div class="evidence">
                    <div class="evidence-text">"{src['snippet']}…"</div>
                    <div class="evidence-ref">Page {src['page']}</div>
                </div>"""
            st.markdown(f"""
            <div class="a-card">
                <div class="a-q">{item['question']}</div>
                <div class="a-sec">Answer</div>
                <div class="a-body">{parsed['direct']}</div>
                {pts_html}{ev_html}
                <div class="a-time">{item.get('time','')}</div>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.thinking:
            st.markdown("""
            <div class="thinking">
                <div class="t-dot"></div>
                <div class="t-text">Analysing document</div>
            </div>
            """, unsafe_allow_html=True)

        question = st.session_state.pending_q or (user_input if st.session_state.q_count < MAX_Q else None)

        if question and not st.session_state.thinking and st.session_state.q_count < MAX_Q:
            st.session_state.pending_q = None
            st.session_state.last_q    = question
            st.session_state.thinking  = True
            st.session_state.q_count  += 1
            st.rerun()

        if st.session_state.thinking and st.session_state.last_q:
            q = st.session_state.last_q
            try:
                result  = st.session_state.chain.invoke({"question": q})
                sources = get_sources(result.get("source_documents", []))
                st.session_state.answers.append({
                    "question": q,
                    "content":  result["answer"],
                    "sources":  sources,
                    "time":     datetime.datetime.now().strftime("%H:%M"),
                    "error":    False,
                })
            except Exception as e:
                err = str(e).lower()
                msg = ("Rate limit — wait 60 seconds." if "rate" in err
                       else "API key error — verify Streamlit secrets." if "api" in err or "key" in err
                       else f"Error: {e}")
                st.session_state.answers.append({"error": True, "content": msg})
            finally:
                st.session_state.thinking = False
                st.session_state.last_q   = None
                st.rerun()

        if st.session_state.q_count >= MAX_Q:
            st.markdown("""
            <div class="limit-wall">
                <div style="font-family:'Playfair Display',serif;font-size:.95rem;
                            color:#4E5768;margin-bottom:.3rem;font-weight:400">
                    Session limit reached
                </div>
                <div style="font-size:.72rem;color:#272F3E;
                            font-family:'JetBrains Mono',monospace;letter-spacing:.04em">
                    10 / 10 questions used · refresh to begin a new session
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.answers:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("Clear answers", key="clear_btn"):
                st.session_state.answers  = []
                st.session_state.thinking = False
                st.session_state.last_q   = None
                if st.session_state.chain:
                    st.session_state.chain.memory.clear()
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
