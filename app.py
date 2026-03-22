import os, time, datetime, tempfile, base64
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ═══════════════════════════════════════════════════════════════════════════════
# SCHOLARLY TYPOGRAPHY & THEME
# ═══════════════════════════════════════════════════════════════════════════════
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

st.set_page_config(page_title="Scriptorium", page_icon="📜", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400..800;1,400..800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-paper: #F7F4EF;
    --sidebar-stone: #EDE9E1;
    --clay-accent: #B06B50;
    --ink-deep: #1A1A1A;
    --ink-muted: #5F5F5F;
    --border-line: rgba(0,0,0,0.06);
}

.stApp { background-color: var(--bg-paper); color: var(--ink-deep); }
header, footer, #MainMenu { visibility: hidden; }

/* Typography Overrides */
h1, h2, h3, .serif-font { 
    font-family: 'EB Garamond', serif !important; 
    font-weight: 500; 
    letter-spacing: -0.01em;
}

div, p, span, button, input { 
    font-family: 'Inter', sans-serif !important; 
}

/* Sidebar Styling */
section[data-testid="stSidebar"] { 
    background-color: var(--sidebar-stone) !important; 
    border-right: 1px solid var(--border-line) !important;
}

/* Chat & Content Bubbles */
.user-query {
    padding: 0.8rem 1.2rem;
    margin-bottom: 2rem;
    background: #FFFFFF;
    border: 1px solid var(--border-line);
    border-radius: 12px;
    font-size: 0.95rem;
}

.ai-response {
    margin-bottom: 3.5rem;
    line-height: 1.7;
}

.label-mini {
    color: var(--clay-accent);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.reading-list-item {
    font-size: 0.85rem;
    padding: 10px;
    border-bottom: 1px solid var(--border-line);
    color: var(--ink-muted);
}

/* PDF Component */
.pdf-frame { 
    border-radius: 4px; 
    border: 1px solid var(--border-line);
    box-shadow: 0 4px 20px rgba(0,0,0,0.03);
}

/* UI Adjustments */
.stButton>button {
    border-radius: 4px !important;
    background: transparent !important;
    color: var(--ink-muted) !important;
    border: 1px solid var(--border-line) !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STATE & LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
if "answers" not in st.session_state:
    st.session_state.update({
        "answers": [], "chain": None, "doc_name": "Untitled", 
        "pdf_b64": None, "abstract": None, "current_page": 1, 
        "reading_list": []
    })

@st.cache_resource(show_spinner=False)
def process_document(_pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_pdf_bytes)
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.unlink(tmp_path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(pages)
    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1), 
        retriever=vectorstore.as_retriever(), 
        return_source_documents=True,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    )
    return chain

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE UI
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h2 class='serif-font' style='font-size:2.2rem;'>Scriptorium</h2>", unsafe_allow_html=True)
    if st.session_state.chain:
        st.markdown(f"<p style='color:var(--ink-muted); font-size:0.85rem;'>{st.session_state.doc_name}</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<div class='label-mini'>Reading List</div>", unsafe_allow_html=True)
        if not st.session_state.reading_list:
            st.markdown("<p style='font-size:0.8rem; font-style:italic; color:var(--ink-muted);'>No saved insights yet.</p>", unsafe_allow_html=True)
        for idx, item in enumerate(st.session_state.reading_list):
            st.markdown(f"<div class='reading-list-item'>{item[:80]}...</div>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("Clear All Data", use_container_width=True):
            st.session_state.clear()
            st.rerun()

if st.session_state.chain is None:
    st.markdown("<div style='height:30vh;'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<h2 class='serif-font' style='text-align:center;'>Begin a new inquiry.</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
        if uploaded_file:
            pdf_bytes = uploaded_file.read()
            with st.status("Reading...", expanded=True):
                chain = process_document(pdf_bytes)
                summary = chain.invoke({"question": "Summarize the core thesis in two sentences."})["answer"]
            st.session_state.update({
                "chain": chain, "doc_name": uploaded_file.name, "abstract": summary,
                "pdf_b64": base64.b64encode(pdf_bytes).decode("utf-8")
            })
            st.rerun()
else:
    col_pdf, col_qa = st.columns([0.45, 0.55])
    
    with col_pdf:
        pdf_url = f"data:application/pdf;base64,{st.session_state.pdf_b64}#page={st.session_state.current_page}"
        st.markdown(f'<iframe src="{pdf_url}" width="100%" height="880vh" class="pdf-frame"></iframe>', unsafe_allow_html=True)

    with col_qa:
        chat_container = st.container(height=780, border=False)
        with chat_container:
            # The Abstract
            st.markdown(f"<div class='ai-response'><div class='label-mini'>Thesis Summary</div><div class='serif-font' style='font-size:1.25rem;'>{st.session_state.abstract}</div></div>", unsafe_allow_html=True)
            
            for i, item in enumerate(st.session_state.answers):
                st.markdown(f"<div class='user-query'>{item['question']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='ai-response'><div class='label-mini'>Analysis</div><div class='serif-font' style='font-size:1.15rem;'>{item['content']}</div><br><small style='color:var(--ink-muted);'>Source: Page {item['page']}</small></div>", unsafe_allow_html=True)
                
                if st.button("Save to Reading List", key=f"save_{i}"):
                    st.session_state.reading_list.append(item['content'])
                    st.toast("Saved to List")
                    st.rerun()

        query = st.chat_input("Inquire further...")
        if query:
            result = st.session_state.chain.invoke({"question": query})
            pg = result["source_documents"][0].metadata.get("page", 0) + 1
            st.session_state.answers.append({"question": query, "content": result["answer"], "page": pg})
            st.session_state.current_page = pg
            st.rerun()
