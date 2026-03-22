"""
ChemInsight AI
Built by Ashlesha Sharma — BSc (H) Chemistry, Miranda House, Delhi University.

A RAG (Retrieval-Augmented Generation) system that lets researchers
upload chemistry papers and ask questions in plain English.

API key loads from Streamlit secrets (deployed) or .env (local).
Session limit: 10 questions per visitor.
"""

import os, time, datetime
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

import tempfile

load_dotenv()

# ── API key: Streamlit secrets (deployed) or .env (local) ─────────────────────
def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")

OPENAI_API_KEY = get_api_key()
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_QUESTIONS = 10
MODEL         = "gpt-4o-mini"
EMBED_MODEL   = "text-embedding-3-small"
CHUNK_SIZE    = 900
CHUNK_OVERLAP = 180
TOP_K         = 4

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChemInsight AI",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: #080d1a !important;
    color: #c9d1e0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 100% !important; }
section[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1a2340 !important;
}

.hero {
    background: linear-gradient(135deg, #0d1a3a 0%, #0a1628 60%, #071020 100%);
    border: 1px solid #1a2d50;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: 'C₆H₁₂O₆  ·  NH₃  ·  H₂SO₄  ·  NaCl  ·  CH₃COOH  ·  C₂H₅OH  ·  KMnO₄  ·  HCl';
    position: absolute;
    top: 14px; right: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #00d4aa18;
    letter-spacing: 0.15em;
    white-space: nowrap;
}
.hero-badge {
    display: inline-block;
    background: #00d4aa15;
    border: 1px solid #00d4aa35;
    color: #00d4aa;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4aa, #4fc3f7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.hero-sub { font-size: 0.98rem; color: #4a5568; margin: 0; }

.stat-row { display: flex; gap: 10px; margin: 1rem 0; }
.stat-card {
    flex: 1;
    background: #0d1628;
    border: 1px solid #1a2d50;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #00d4aa44; }
.stat-num {
    font-size: 1.8rem;
    font-weight: 700;
    color: #00d4aa;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.stat-label {
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

.usage-wrap {
    background: #0d1628;
    border: 1px solid #1a2d50;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 0.5rem 0;
}
.usage-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #475569;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.usage-track { height: 5px; background: #1a2d50; border-radius: 3px; overflow: hidden; }
.usage-fill  { height: 100%; border-radius: 3px; transition: width 0.4s ease; }

.status-ready {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #00d4aa13;
    border: 1px solid #00d4aa35;
    color: #00d4aa;
    padding: 5px 13px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-bottom: 10px;
}
.pulse {
    width: 7px; height: 7px;
    background: #00d4aa;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.4; transform:scale(.75); }
}

.sec {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2d3f5a;
    margin: 1.2rem 0 0.55rem;
    font-family: 'JetBrains Mono', monospace;
}

.bubble-user {
    max-width: 78%;
    background: linear-gradient(135deg,#1a3a6e,#1e4080);
    border: 1px solid #2a5099;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    font-size: 0.95rem;
    color: #c9d8f0;
    position: relative;
    margin-left: auto;
}
.bubble-user::before {
    content: '🧪';
    font-size: 0.7rem;
    position: absolute;
    top: -9px; right: 8px;
    background: #1a3a6e;
    padding: 1px 5px;
    border-radius: 6px;
    border: 1px solid #2a5099;
}
.bubble-ai {
    max-width: 88%;
    background: #0d1628;
    border: 1px solid #1a2d50;
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    font-size: 0.95rem;
    color: #c9d1e0;
    line-height: 1.7;
    position: relative;
}
.bubble-ai::before {
    content: '⚗️';
    font-size: 0.7rem;
    position: absolute;
    top: -9px; left: 8px;
    background: #0d1628;
    padding: 1px 5px;
    border-radius: 6px;
    border: 1px solid #1a2d50;
}
.btime {
    font-size: 0.67rem;
    color: #2d3f5a;
    margin-top: 5px;
    font-family: 'JetBrains Mono', monospace;
}

.src-pill {
    display: inline-block;
    background: #fbbf2413;
    border: 1px solid #fbbf2435;
    color: #fbbf24;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 12px;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}
.src-text {
    background: #09111e;
    border-left: 3px solid #fbbf24;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #475569;
    margin: 7px 0 0;
    font-style: italic;
    line-height: 1.55;
    font-family: 'JetBrains Mono', monospace;
}

.empty-state {
    background: #0d1628;
    border: 1px solid #1a2d50;
    border-radius: 14px;
    padding: 3rem 2rem;
    text-align: center;
    margin-top: 1rem;
}
.limit-wall {
    background: linear-gradient(135deg,#1a1028,#0d0a1a);
    border: 1px solid #4a1c6e;
    border-radius: 14px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1rem;
}

.stButton > button {
    background: #0d1628 !important;
    border: 1px solid #1a2d50 !important;
    color: #64748b !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all .2s !important;
    width: 100%;
    text-align: left !important;
    justify-content: flex-start !important;
}
.stButton > button:hover {
    border-color: #00d4aa55 !important;
    color: #00d4aa !important;
    background: #0d1a28 !important;
}
div[data-testid="stChatInput"] textarea {
    background: #0d1628 !important;
    border: 1px solid #1a2d50 !important;
    color: #c9d1e0 !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 2px #00d4aa18 !important;
}
.streamlit-expanderHeader {
    background: #0d1628 !important;
    border: 1px solid #1a2d50 !important;
    border-radius: 8px !important;
    color: #475569 !important;
    font-size: 0.8rem !important;
}
.stProgress > div > div { background: #00d4aa !important; }
div[data-testid="stFileUploader"] {
    background: #0d1628 !important;
    border: 2px dashed #1a2d50 !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "messages":  [],
    "chain":     None,
    "doc_name":  None,
    "doc_stats": None,
    "q_count":   0,
    "pending_q": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-size:1.3rem;font-weight:700;color:#00d4aa;letter-spacing:-.02em">
        ⚗️ ChemInsight AI
    </div>
    <div style="font-size:.78rem;color:#2d3f5a;margin-bottom:1.2rem">
        Research papers, finally searchable.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    used  = st.session_state.q_count
    left  = MAX_QUESTIONS - used
    pct   = int((used / MAX_QUESTIONS) * 100)
    color = "#00d4aa" if pct < 60 else "#fbbf24" if pct < 90 else "#f87171"

    st.markdown(f"""
    <div class="usage-wrap">
        <div class="usage-label">
            <span>Session usage</span>
            <span style="color:{color}">{used}/{MAX_QUESTIONS} questions</span>
        </div>
        <div class="usage-track">
            <div class="usage-fill" style="width:{pct}%;background:{color}"></div>
        </div>
        <div style="font-size:.72rem;color:#2d3f5a;margin-top:6px">
            {left} question{'s' if left != 1 else ''} remaining · refresh to reset
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sec">How RAG works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.8rem;color:#334155;line-height:1.8">
        <span style="color:#00d4aa">1.</span> PDF → split into chunks<br>
        <span style="color:#00d4aa">2.</span> Chunks → number vectors<br>
        <span style="color:#00d4aa">3.</span> Question → find closest vectors<br>
        <span style="color:#00d4aa">4.</span> Closest chunks + question → GPT<br>
        <span style="color:#00d4aa">5.</span> Answer grounded in <em>your doc</em><br><br>
        GPT answers <span style="color:#fbbf24;font-weight:500">only</span>
        from your paper — not the internet.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="font-size:.75rem;color:#2d3f5a;line-height:1.9">
        Built by
        <span style="color:#94a3b8;font-weight:500">Ashlesha Sharma</span><br>
        BSc (H) Chemistry<br>
        Miranda House · Delhi University<br><br>
        <span style="color:#00d4aa">→</span>
        <a href="https://github.com/ashlesha-sharma/cheminsight-ai"
           style="color:#4fc3f7;text-decoration:none">
           github.com/ashlesha-sharma
        </a>
    </div>
    """, unsafe_allow_html=True)


# ── API key guard ──────────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ No API key found. Add OPENAI_API_KEY to Streamlit secrets (deployed) or a .env file (local).")
    st.stop()


# ── Core functions ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_chain(_pdf_bytes: bytes, filename: str):
    """
    Processes a PDF and returns a conversational RAG chain + document stats.
    Cached — runs only once per unique file, even if the page reruns.
    """
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
        persist_directory=f"./chroma_{filename[:18].replace('.','_')}",
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
        template="""You are ChemInsight AI — a specialist chemistry research assistant.

Your job is to answer questions about a specific scientific paper uploaded by the user.

Rules:
- Answer ONLY from the context below. If the answer is not there, say: "This specific detail isn't covered in the uploaded paper."
- Never guess or use outside knowledge.
- Always quote specific numbers: yields, temperatures, concentrations, wavelengths, times.
- For reaction mechanisms, explain step by step.
- Be precise but clear — the user may be a student trying to understand, not just extract data.

Context from the paper:
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
            snippet = doc.page_content[:190].replace("\n", " ").strip()
            out.append({"page": page + 1, "snippet": snippet})
    return out


def density_chart(page_lengths):
    avg    = sum(page_lengths) / len(page_lengths)
    colors = ["#00d4aa" if l >= avg else "#1a2d50" for l in page_lengths]
    fig    = go.Figure(go.Bar(
        x=list(range(1, len(page_lengths) + 1)),
        y=page_lengths,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="Page %{x} — %{y} characters<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Text density · teal = above average",
                   font=dict(size=11, color="#334155"), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit", color="#475569", size=10),
        margin=dict(l=10, r=10, t=36, b=28),
        height=175,
        xaxis=dict(showgrid=False, zeroline=False,
                   title="Page", title_font_size=10,
                   tickfont_size=9, color="#334155"),
        yaxis=dict(showgrid=True, gridcolor="#0d1628",
                   zeroline=False, title="Chars",
                   title_font_size=10, tickfont_size=9, color="#334155"),
        bargap=0.25,
    )
    return fig


def donut_chart(chunks, pages):
    fig = go.Figure(go.Pie(
        values=[chunks, max(1, pages)],
        labels=["Chunks indexed", "Pages"],
        hole=0.65,
        marker=dict(colors=["#00d4aa", "#1a2d50"], line=dict(width=0)),
        textinfo="none",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{chunks}</b><br><span style='font-size:9px'>chunks</span>",
        showarrow=False,
        font=dict(color="#c9d1e0", size=15),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=5, r=5, t=5, b=5),
        height=150,
    )
    return fig


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚗ RAG · LangChain · OpenAI · Free to use</div>
    <div class="hero-title">ChemInsight AI</div>
    <div class="hero-sub">
        Upload any chemistry paper. Ask questions. Get answers grounded in the actual
        document — not the internet, not guesswork.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.6], gap="large")


# ══ LEFT — Upload + stats ══════════════════════════════════════════════════════
with left:
    st.markdown('<div class="sec">📄 Upload your paper</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a chemistry PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Research papers, textbook chapters, lab reports. Needs selectable text.",
    )

    if uploaded:
        if st.session_state.doc_name != uploaded.name:
            st.session_state.messages = []
            st.session_state.q_count  = 0

            bar = st.progress(0, text="Reading PDF...")
            time.sleep(0.2)
            bar.progress(25, text="Splitting into chunks...")
            time.sleep(0.1)
            bar.progress(50, text="Building vector index — takes about 20 seconds...")

            try:
                chain, stats = build_chain(uploaded.read(), uploaded.name)
                bar.progress(90, text="Almost done...")
                time.sleep(0.3)
                bar.progress(100, text="Ready!")
                time.sleep(0.4)
                bar.empty()

                st.session_state.chain     = chain
                st.session_state.doc_name  = uploaded.name
                st.session_state.doc_stats = stats

            except ValueError as e:
                bar.empty()
                if "SCANNED_PDF" in str(e):
                    st.error("⚠️ This PDF appears to be scanned — no extractable text found. Try a PDF where you can select and copy text.")
                else:
                    st.error(f"Error: {e}")
            except Exception as e:
                bar.empty()
                err = str(e).lower()
                if "api" in err or "key" in err or "auth" in err:
                    st.error("API key error. Check that the key is set correctly in Streamlit secrets.")
                elif "rate" in err:
                    st.error("Rate limit hit — wait 60 seconds and try again.")
                else:
                    st.error(f"Something went wrong: {e}")

        if st.session_state.doc_stats:
            s    = st.session_state.doc_stats
            name = st.session_state.doc_name
            display_name = name[:38] + "…" if len(name) > 38 else name

            st.markdown(f"""
            <div class="status-ready">
                <span class="pulse"></span>
                {display_name}
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
                    <div class="stat-label">Chunks</div>
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

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    donut_chart(s["chunks"], s["pages"]),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with c2:
                avg         = sum(s["page_lengths"]) / len(s["page_lengths"])
                dense_pages = sum(1 for l in s["page_lengths"] if l > avg)
                st.markdown(f"""
                <div style="padding:14px 0;font-size:.8rem;color:#334155;line-height:2.1">
                    <span style="color:#00d4aa">●</span>
                    {dense_pages} dense pages<br>
                    <span style="color:#1a2d50">●</span>
                    {s['pages'] - dense_pages} lighter pages<br>
                    <span style="color:#fbbf24">avg</span>
                    {int(avg):,} chars/pg<br>
                    <span style="color:#475569">total</span>
                    {s['chars']:,} chars
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="sec">💡 Quick questions</div>', unsafe_allow_html=True)
            for q in [
                "🎯  What is the main objective?",
                "⚗️  What reagents and conditions were used?",
                "📊  What were the key results and yields?",
                "🔬  Explain the proposed mechanism.",
                "⚠️  What limitations are mentioned?",
                "📝  Summarise the experimental procedure.",
            ]:
                if st.button(q, key=f"q_{q}"):
                    st.session_state.pending_q = q[3:].strip()
                    st.rerun()

    else:
        st.markdown("""
        <div style="
            background:#0d1628; border:2px dashed #1a2d50;
            border-radius:14px; padding:2.5rem; text-align:center;
        ">
            <div style="font-size:2rem;margin-bottom:.5rem">📄</div>
            <div style="font-size:1rem;color:#475569;font-weight:500;margin-bottom:.3rem">
                Drop a chemistry PDF here
            </div>
            <div style="font-size:.82rem;color:#2d3f5a">
                Research paper · Textbook chapter · Lab report
            </div>
        </div>
        <div style="margin-top:14px;font-size:.8rem;color:#2d3f5a;line-height:2.1">
            <span style="color:#00d4aa">→</span> RSC / ACS / Elsevier papers work great<br>
            <span style="color:#00d4aa">→</span> Must have selectable text — not scanned<br>
            <span style="color:#00d4aa">→</span> Up to ~50 pages works well<br>
            <span style="color:#fbbf24">→</span> Free to use · No signup needed
        </div>
        """, unsafe_allow_html=True)


# ══ RIGHT — Chat ═══════════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="sec">💬 Ask about the paper</div>', unsafe_allow_html=True)

    if st.session_state.chain is None:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:2rem;margin-bottom:.8rem">🔬</div>
            <div style="font-size:1rem;color:#334155;font-weight:500">
                Upload a paper on the left to begin
            </div>
            <div style="font-size:.82rem;color:#2d3f5a;margin-top:.4rem">
                Then ask anything — yields, mechanisms, conditions, findings
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        for msg in st.session_state.messages:
            ts = msg.get("time", "")
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="bubble-user">
                    {msg["content"]}
                    <div class="btime">{ts}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                body = msg["content"].replace("\n", "<br>")
                st.markdown(f"""
                <div class="bubble-ai">
                    {body}
                    <div class="btime">{ts}</div>
                </div>
                """, unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander(
                        f"📚 {len(msg['sources'])} source passage{'s' if len(msg['sources'])>1 else ''} used"
                    ):
                        for src in msg["sources"]:
                            st.markdown(f"""
                            <span class="src-pill">page {src['page']}</span>
                            <div class="src-text">"{src['snippet']}…"</div>
                            """, unsafe_allow_html=True)

        st.divider()

        if st.session_state.q_count >= MAX_QUESTIONS:
            st.markdown("""
            <div class="limit-wall">
                <div style="font-size:1.8rem;margin-bottom:.8rem">🔒</div>
                <div style="font-size:1rem;color:#a78bfa;font-weight:600;margin-bottom:.4rem">
                    Session limit reached
                </div>
                <div style="font-size:.85rem;color:#475569;line-height:1.7">
                    You have used all 10 questions for this session.<br>
                    Refresh the page to start a new session.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            remaining  = MAX_QUESTIONS - st.session_state.q_count
            user_input = st.chat_input(
                f"Ask anything about the paper — {remaining} question{'s' if remaining != 1 else ''} left"
            )
            question = st.session_state.pending_q or user_input

            if question:
                st.session_state.pending_q = None
                now = datetime.datetime.now().strftime("%H:%M")

                st.session_state.messages.append({
                    "role": "user", "content": question, "time": now,
                })
                st.session_state.q_count += 1

                with st.spinner("⚗️ Reading the paper…"):
                    try:
                        result  = st.session_state.chain.invoke({"question": question})
                        answer  = result["answer"]
                        sources = get_sources(result.get("source_documents", []))
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "time": datetime.datetime.now().strftime("%H:%M"),
                        })
                    except Exception as e:
                        err = str(e).lower()
                        if "rate" in err:
                            msg = "Rate limit hit — wait 60 seconds and try again."
                        elif "api" in err or "key" in err:
                            msg = "API key error — contact the app owner."
                        else:
                            msg = f"Unexpected error: {e}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"⚠️ {msg}",
                            "sources": [],
                            "time": datetime.datetime.now().strftime("%H:%M"),
                        })

                st.rerun()

            if st.session_state.messages:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑 Clear conversation"):
                    st.session_state.messages = []
                    if st.session_state.chain:
                        st.session_state.chain.memory.clear()
                    st.rerun()
