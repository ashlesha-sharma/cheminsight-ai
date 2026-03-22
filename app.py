"""
ChemInsight AI 
Built by Ashlesha Sharma, BSc (H) Chemistry, Miranda House, Delhi University.
Old-money aesthetic. Structured answer cards. RAG pipeline.
"""

import os, time, datetime, tempfile
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
    page_title="ChemInsight AI",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400;1,500&family=Inter:wght@300;400;500&display=swap');

/* ── Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #F5F0E8 !important;
    color: #2C2416 !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #EDE6D6 !important;
    border-right: 1px solid #D4C9B0 !important;
}

/* ── Typography ── */
.serif { font-family: 'Playfair Display', serif !important; }

/* ── Hero ── */
.hero {
    background: #FEFCF7;
    border: 1px solid #D4C9B0;
    border-radius: 3px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: #B8965A;
}
.hero-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #B8965A;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 500;
    color: #1C1810;
    line-height: 1.15;
    margin: 0 0 0.6rem 0;
    letter-spacing: -0.01em;
}
.hero-sub {
    font-size: 0.95rem;
    color: #7A6E5C;
    line-height: 1.7;
    max-width: 560px;
    font-weight: 300;
}

/* ── Section label ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #9E8E72;
    margin: 1.5rem 0 0.7rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #D4C9B0;
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 8px; margin: 1rem 0; }
.stat-card {
    flex: 1;
    background: #FEFCF7;
    border: 1px solid #D4C9B0;
    border-radius: 2px;
    padding: 12px 14px;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #B8965A; }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 500;
    color: #1C1810;
    line-height: 1;
}
.stat-label {
    font-size: 0.65rem;
    color: #9E8E72;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 3px;
}

/* ── Usage bar ── */
.usage-wrap {
    background: #FEFCF7;
    border: 1px solid #D4C9B0;
    border-radius: 2px;
    padding: 12px 16px;
    margin: 0.5rem 0;
}
.usage-top {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    margin-bottom: 7px;
}
.usage-left { color: #9E8E72; letter-spacing: 0.06em; text-transform: uppercase; font-size: 0.65rem; }
.usage-right { font-family: 'Playfair Display', serif; font-size: 0.85rem; color: #2C2416; }
.usage-track {
    height: 2px;
    background: #D4C9B0;
    border-radius: 1px;
    overflow: hidden;
}
.usage-fill {
    height: 100%;
    border-radius: 1px;
    transition: width 0.5s ease;
}

/* ── Ready pill ── */
.status-ready {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #F0EAD8;
    border: 1px solid #D4C9B0;
    color: #7A6E5C;
    padding: 5px 13px;
    border-radius: 20px;
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    margin-bottom: 10px;
}
.dot-gold {
    width: 6px; height: 6px;
    background: #B8965A;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:.4; }
}

/* ── Quick question buttons ── */
.stButton > button {
    background: #FEFCF7 !important;
    border: 1px solid #D4C9B0 !important;
    color: #7A6E5C !important;
    border-radius: 2px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    transition: all .2s !important;
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    padding: 8px 14px !important;
}
.stButton > button:hover {
    border-color: #B8965A !important;
    color: #B8965A !important;
    background: #FAF6EE !important;
}

/* ── Answer cards ── */
.answer-card {
    background: #FEFCF7;
    border: 1px solid #D4C9B0;
    border-radius: 2px;
    padding: 24px 28px;
    margin: 16px 0;
    position: relative;
}
.answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #B8965A;
    opacity: 0.6;
}
.answer-question {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-style: italic;
    color: #5C4F3A;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #E8E0D0;
    font-weight: 400;
}
.answer-section-head {
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #B8965A;
    margin: 16px 0 6px 0;
}
.answer-body {
    font-size: 0.92rem;
    color: #3A3020;
    line-height: 1.75;
    font-weight: 300;
}
.answer-body strong { font-weight: 500; color: #2C2416; }
.answer-time {
    font-size: 0.65rem;
    color: #C4B89A;
    margin-top: 14px;
    letter-spacing: 0.06em;
    text-align: right;
}

/* ── Evidence block ── */
.evidence-block {
    background: #F5F0E8;
    border-left: 2px solid #D4C9B0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.83rem;
    color: #7A6E5C;
    font-style: italic;
    line-height: 1.6;
    border-radius: 0 2px 2px 0;
}
.evidence-page {
    font-style: normal;
    font-size: 0.65rem;
    color: #B8965A;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Empty state ── */
.empty-state {
    background: #FEFCF7;
    border: 1px solid #D4C9B0;
    border-radius: 2px;
    padding: 4rem 2rem;
    text-align: center;
}
.empty-icon {
    font-size: 2rem;
    margin-bottom: 1rem;
    opacity: 0.4;
}
.empty-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #7A6E5C;
    margin-bottom: 0.4rem;
    font-weight: 400;
}
.empty-sub {
    font-size: 0.82rem;
    color: #A89880;
    font-weight: 300;
}

/* ── Limit wall ── */
.limit-wall {
    background: #FEFCF7;
    border: 1px solid #D4C9B0;
    border-radius: 2px;
    padding: 2.5rem;
    text-align: center;
}

/* ── Processing shimmer ── */
.shimmer-wrap {
    padding: 20px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.shimmer-line {
    height: 1px;
    flex: 1;
    background: linear-gradient(90deg, #D4C9B0 0%, #B8965A 50%, #D4C9B0 100%);
    background-size: 200% 100%;
    animation: shimmer 1.8s ease-in-out infinite;
}
@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}
.shimmer-text {
    font-size: 0.75rem;
    color: #9E8E72;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] textarea {
    background: #FEFCF7 !important;
    border: 1px solid #D4C9B0 !important;
    color: #2C2416 !important;
    border-radius: 2px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #B8965A !important;
    box-shadow: 0 0 0 2px #B8965A18 !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
    background: #FEFCF7 !important;
    border: 1px dashed #C4B89A !important;
    border-radius: 2px !important;
}

/* ── Streamlit overrides ── */
.stProgress > div > div { background: #B8965A !important; }
.streamlit-expanderHeader {
    background: #FEFCF7 !important;
    border: 1px solid #D4C9B0 !important;
    border-radius: 2px !important;
    color: #7A6E5C !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em !important;
}
div[data-testid="stExpander"] {
    border: none !important;
}

/* ── Divider ── */
hr { border-color: #D4C9B0 !important; opacity: 0.6 !important; }

/* ── Sidebar text overrides ── */
.sidebar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    font-weight: 500;
    color: #1C1810;
    letter-spacing: -0.01em;
    margin-bottom: 2px;
}
.sidebar-tagline {
    font-size: 0.75rem;
    color: #9E8E72;
    font-weight: 300;
    letter-spacing: 0.04em;
    margin-bottom: 1.2rem;
}
.sidebar-section {
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #B8965A;
    margin: 1.2rem 0 0.6rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "answers":   [],
    "chain":     None,
    "doc_name":  None,
    "doc_stats": None,
    "q_count":   0,
    "pending_q": None,
    "thinking":  False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">⚗ ChemInsight</div>
    <div class="sidebar-tagline">Research papers, finally searchable.</div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="margin:0.5rem 0">', unsafe_allow_html=True)

    used  = st.session_state.q_count
    left  = MAX_QUESTIONS - used
    pct   = int((used / MAX_QUESTIONS) * 100)
    color = "#B8965A" if pct < 60 else "#C4845A" if pct < 90 else "#A85A5A"

    st.markdown(f"""
    <div class="usage-wrap">
        <div class="usage-top">
            <span class="usage-left">Session usage</span>
            <span class="usage-right">{used} / {MAX_QUESTIONS}</span>
        </div>
        <div class="usage-track">
            <div class="usage-fill" style="width:{pct}%;background:{color}"></div>
        </div>
        <div style="font-size:.68rem;color:#A89880;margin-top:6px;letter-spacing:.04em">
            {left} question{'s' if left != 1 else ''} remaining · refresh to reset
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.8rem;color:#7A6E5C;line-height:1.9;font-weight:300">
        <span style="color:#B8965A">i.</span>&nbsp; PDF → split into chunks<br>
        <span style="color:#B8965A">ii.</span>&nbsp; Chunks → embedding vectors<br>
        <span style="color:#B8965A">iii.</span>&nbsp; Question → nearest vectors<br>
        <span style="color:#B8965A">iv.</span>&nbsp; Chunks + question → GPT<br>
        <span style="color:#B8965A">v.</span>&nbsp; Answer from <em>your document only</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="margin:1rem 0">', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:.72rem;color:#A89880;line-height:1.9;font-weight:300">
        Built by
        <span style="color:#5C4F3A;font-weight:400">Ashlesha Sharma</span><br>
        BSc (H) Chemistry<br>
        Miranda House · Delhi University<br><br>
        <a href="https://github.com/ashlesha-sharma/cheminsight-ai"
           style="color:#B8965A;text-decoration:none;font-size:.7rem;letter-spacing:.04em">
           github.com/ashlesha-sharma ↗
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
        persist_directory=f"./chroma_{filename[:16].replace('.','_')}",
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
        template="""You are ChemInsight AI — a precise, expert chemistry research assistant.

Answer questions about this scientific paper with scholarly precision.

Rules:
- Answer ONLY from the context below. If not present: "This detail is not covered in the uploaded paper."
- Quote exact values: yields, temperatures, concentrations, wavelengths, reaction times.
- For mechanisms: explain each step clearly and in order.
- Structure your answer in three parts:
  1. A direct answer (1–3 sentences)
  2. Key points (3–5 bullet points if applicable)
  3. A direct quote or evidence from the paper

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


def density_chart(page_lengths):
    avg    = sum(page_lengths) / len(page_lengths)
    colors = ["#B8965A" if l >= avg else "#D4C9B0" for l in page_lengths]
    fig    = go.Figure(go.Bar(
        x=list(range(1, len(page_lengths) + 1)),
        y=page_lengths,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="Page %{x} — %{y} chars<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Text density per page",
                   font=dict(size=11, color="#9E8E72",
                             family="Inter"), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#9E8E72", size=10),
        margin=dict(l=10, r=10, t=36, b=28),
        height=160,
        xaxis=dict(showgrid=False, zeroline=False,
                   title="Page", title_font_size=10,
                   tickfont_size=9, color="#9E8E72",
                   linecolor="#D4C9B0"),
        yaxis=dict(showgrid=True, gridcolor="#EDE6D6",
                   zeroline=False,
                   tickfont_size=9, color="#9E8E72"),
        bargap=0.3,
    )
    return fig


def parse_answer(raw: str):
    """
    Parse GPT answer into structured sections.
    Returns dict with: direct, points, evidence
    """
    lines  = raw.strip().split("\n")
    direct = []
    points = []
    evidence = []
    mode = "direct"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if any(k in low for k in ["key point", "key finding", "important point", "notable"]):
            mode = "points"
            continue
        if any(k in low for k in ["evidence", "quote", "from the paper", "according to", "the paper state"]):
            mode = "evidence"
            continue
        if stripped.startswith("•") or stripped.startswith("-") or stripped.startswith("*"):
            points.append(stripped.lstrip("•-* "))
            mode = "points"
        elif stripped.startswith('"') or stripped.startswith("'"):
            evidence.append(stripped.strip('"\''))
            mode = "evidence"
        else:
            if mode == "direct":
                direct.append(stripped)
            elif mode == "points":
                points.append(stripped)
            elif mode == "evidence":
                evidence.append(stripped)

    return {
        "direct":   " ".join(direct) if direct else raw.strip(),
        "points":   points[:6],
        "evidence": evidence[:2],
    }


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">RAG · LangChain · OpenAI · Free to use</div>
    <div class="hero-title">ChemInsight AI</div>
    <div class="hero-sub">
        Upload any chemistry paper. Ask questions in plain English.<br>
        Every answer is grounded in your document — not the internet, not guesswork.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.55], gap="large")


# ══ LEFT ═══════════════════════════════════════════════════════════════════════
with left:

    st.markdown("""
    <div class="section-label">Upload document</div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Research papers, textbook chapters, lab reports. Needs selectable text.",
    )

    if uploaded:
        if st.session_state.doc_name != uploaded.name:
            st.session_state.answers  = []
            st.session_state.q_count  = 0

            bar = st.progress(0, text="Reading document...")
            time.sleep(0.2)
            bar.progress(25, text="Splitting into passages...")
            time.sleep(0.1)
            bar.progress(50, text="Building index — ~20 seconds...")

            try:
                chain, stats = build_chain(uploaded.read(), uploaded.name)
                bar.progress(90, text="Finalising...")
                time.sleep(0.3)
                bar.progress(100, text="Ready.")
                time.sleep(0.4)
                bar.empty()

                st.session_state.chain     = chain
                st.session_state.doc_name  = uploaded.name
                st.session_state.doc_stats = stats

            except ValueError as e:
                bar.empty()
                if "SCANNED_PDF" in str(e):
                    st.error("This PDF appears to be scanned — no selectable text found. Try a PDF where text can be highlighted.")
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

        if st.session_state.doc_stats:
            s    = st.session_state.doc_stats
            name = st.session_state.doc_name
            dn   = name[:40] + "…" if len(name) > 40 else name

            st.markdown(f"""
            <div class="status-ready">
                <span class="dot-gold"></span>
                {dn}
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
                    <div class="stat-label">Est. read</div>
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

            st.markdown("""
            <div class="section-label">Suggested questions</div>
            """, unsafe_allow_html=True)

            for q in [
                "What is the main objective of this study?",
                "What reagents and conditions were used?",
                "What were the key results and yields?",
                "Explain the proposed mechanism.",
                "What limitations does the paper mention?",
                "Summarise the experimental procedure.",
            ]:
                if st.button(q, key=f"sq_{q}"):
                    st.session_state.pending_q = q
                    st.rerun()

    else:
        st.markdown("""
        <div style="
            background:#FEFCF7; border:1px dashed #C4B89A;
            border-radius:2px; padding:3rem 2rem; text-align:center;
        ">
            <div style="font-size:1.8rem;opacity:.3;margin-bottom:.8rem">⚗</div>
            <div style="font-family:'Playfair Display',serif;font-size:1rem;
                        color:#7A6E5C;font-weight:400;margin-bottom:.4rem">
                Drop a chemistry paper here
            </div>
            <div style="font-size:.8rem;color:#A89880;font-weight:300">
                Research paper · Textbook chapter · Lab report
            </div>
        </div>
        <div style="margin-top:14px;font-size:.78rem;color:#A89880;
                    line-height:2.1;font-weight:300">
            <span style="color:#B8965A">→</span>&nbsp;
            RSC / ACS / Elsevier papers work well<br>
            <span style="color:#B8965A">→</span>&nbsp;
            Must have selectable text — not scanned<br>
            <span style="color:#B8965A">→</span>&nbsp;
            Free to use · No account needed
        </div>
        """, unsafe_allow_html=True)


# ══ RIGHT ══════════════════════════════════════════════════════════════════════
with right:
    st.markdown("""
    <div class="section-label">Ask about the paper</div>
    """, unsafe_allow_html=True)

    if st.session_state.chain is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <div class="empty-title">Upload a paper to begin</div>
            <div class="empty-sub">
                Ask about yields, mechanisms, conditions, findings —
                any detail in the document
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Answer cards ──────────────────────────────────────────────────────
        for item in st.session_state.answers:
            if item.get("error"):
                st.error(item["content"])
                continue

            parsed = parse_answer(item["content"])

            points_html = ""
            if parsed["points"]:
                pts = "".join(f"<li style='margin-bottom:4px'>{p}</li>"
                              for p in parsed["points"])
                points_html = f"""
                <div class="answer-section-head">Key Points</div>
                <ul style="margin:0;padding-left:18px;font-size:.88rem;
                           color:#3A3020;line-height:1.7;font-weight:300">
                    {pts}
                </ul>"""

            evidence_html = ""
            if item.get("sources"):
                src = item["sources"][0]
                evidence_html = f"""
                <div class="answer-section-head">Evidence from Paper</div>
                <div class="evidence-block">
                    "{src['snippet']}…"
                    <div class="evidence-page">Page {src['page']}</div>
                </div>"""

            st.markdown(f"""
            <div class="answer-card">
                <div class="answer-question">"{item['question']}"</div>
                <div class="answer-section-head">Answer</div>
                <div class="answer-body">{parsed['direct']}</div>
                {points_html}
                {evidence_html}
                <div class="answer-time">{item.get('time','')}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Thinking state ────────────────────────────────────────────────────
        if st.session_state.thinking:
            st.markdown("""
            <div class="shimmer-wrap">
                <div class="shimmer-line"></div>
                <div class="shimmer-text">Analysing document</div>
                <div class="shimmer-line"></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        # ── Session limit ─────────────────────────────────────────────────────
        if st.session_state.q_count >= MAX_QUESTIONS:
            st.markdown("""
            <div class="limit-wall">
                <div style="font-family:'Playfair Display',serif;font-size:1.1rem;
                            color:#5C4F3A;margin-bottom:.5rem">
                    Session limit reached
                </div>
                <div style="font-size:.82rem;color:#9E8E72;font-weight:300">
                    You have used all 10 questions for this session.<br>
                    Refresh the page to begin a new session.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            remaining  = MAX_QUESTIONS - st.session_state.q_count
            user_input = st.chat_input(
                f"Ask a question about the paper  ·  {remaining} remaining"
            )
            question = st.session_state.pending_q or user_input

            if question:
                st.session_state.pending_q = None
                st.session_state.thinking  = True
                now = datetime.datetime.now().strftime("%H:%M")
                st.session_state.q_count  += 1
                st.rerun()

            # Process if thinking flag is set but no answer yet for this q
            if (st.session_state.thinking and
                    (not st.session_state.answers or
                     st.session_state.answers[-1].get("processed"))):
                st.session_state.thinking = False

        # ── Actually get the answer (runs after rerun) ─────────────────────
        if (st.session_state.thinking and question):
            try:
                result  = st.session_state.chain.invoke({"question": question})
                answer  = result["answer"]
                sources = get_sources(result.get("source_documents", []))
                st.session_state.answers.append({
                    "question":  question,
                    "content":   answer,
                    "sources":   sources,
                    "time":      datetime.datetime.now().strftime("%H:%M"),
                    "processed": True,
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
                    "error":     True,
                    "content":   msg,
                    "processed": True,
                })
            finally:
                st.session_state.thinking = False
                st.rerun()

        # ── Clear ─────────────────────────────────────────────────────────────
        if st.session_state.answers:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("Clear answers", key="clear"):
                st.session_state.answers = []
                if st.session_state.chain:
                    st.session_state.chain.memory.clear()
                st.rerun()
