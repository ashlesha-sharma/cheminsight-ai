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
# GEMINI DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

st.set_page_config(page_title="Scriptorium", page_icon="✨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500&family=Inter:wght@300;400;500&display=swap');

:root {
    --gemini-bg: #131314;
    --gemini-card: #1e1f20;
    --gemini-blue: #4285f4;
    --gemini-gradient: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
    --text-main: #e3e3e3;
    --text-dim: #b4b4b4;
}

.stApp { background-color: var(--gemini-bg); color: var(--text-main); }
header, footer, #MainMenu { visibility: hidden; }

/* Typography */
h1, h2, h3, div, p, span { font-family: 'Inter', sans-serif !important; }

/* Sidebar - Gemini Style */
section[data-testid="stSidebar"] { 
    background-color: #1e1f20 !important; 
    border-right: none !important;
    width: 300px !important;
}

/* Gemini Gradient Text */
.gemini-text {
    background: var(--gemini-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 500;
}

/* Chat Bubbles */
.user-msg {
    padding: 1rem;
    margin-bottom: 1.5rem;
    background: transparent;
    border-radius: 12px;
}

.ai-msg {
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    background: var(--gemini-card);
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Pill Input */
.stChatInputContainer {
    padding: 10px 20px !important;
    background: #1e1f20 !important;
    border-radius: 50px !important;
}

/* PDF View */
.pdf-frame { 
    border-radius: 24px; 
    border: 8px solid #1e1f20;
    filter: brightness(0.9) contrast(1.1);
}

/* Buttons */
.stButton>button {
    border-radius: 50px !important;
    background: #282a2d !important;
    color: white !important;
    border: 1px solid #3c4043 !important;
    transition: 0.3s;
}
.stButton>button:hover {
    background: #3c4043 !important;
    border-color: var(--gemini-blue) !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGIC & EXPORT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
if "answers" not in st.session_state:
    st.session_state.update({
        "answers": [], "chain": None, "doc_name": "Untitled", 
        "pdf_b64": None, "abstract": None, "current_page": 1, "synthesis": None
    })

def create_markdown_report():
    report = f"# Scriptorium Research Report\n"
    report += f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
    report += f"## Document: {st.session_state.doc_name}\n\n"
    report += f"### Executive Abstract\n{st.session_state.abstract}\n\n"
    
    if st.session_state.synthesis:
        report += f"### Synthesized Review\n{st.session_state.synthesis}\n\n"
        
    report += "## Investigation Log\n"
    for item in st.session_state.answers:
        report += f"**Q:** {item['question']}\n"
        report += f"**A:** {item['content']}\n"
        report += f"*(Ref: Page {item['page']})*\n\n"
    return report

@st.cache_resource(show_spinner=False)
def process_document(_pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_pdf_bytes)
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.unlink(tmp_path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(pages)
    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0), 
        retriever=vectorstore.as_retriever(), 
        return_source_documents=True,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    )
    return chain

# ═══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h2 class='gemini-text' style='margin-bottom:0;'>Scriptorium</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.8rem; color:var(--text-dim);'>Powered by Gemini Aesthetic</p>", unsafe_allow_html=True)
    
    if st.session_state.chain:
        st.markdown("---")
        if st.button("✨ Synthesize Review", use_container_width=True):
            chat_log = "\n".join([f"Q: {a['question']} A: {a['content']}" for a in st.session_state.answers])
            st.session_state.synthesis = ChatOpenAI(model_name="gpt-4o-mini").invoke(f"Synthesize this research discussion into a formal paragraph: {chat_log}").content
        
        # EXPORT BUTTON
        md_content = create_markdown_report()
        st.download_button(
            label="📄 Export Report (.md)",
            data=md_content,
            file_name=f"Research_Report_{st.session_state.doc_name.split('.')[0]}.md",
            mime="text/markdown",
            use_container_width=True
        )
        
        if st.button("New Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()

if st.session_state.chain is None:
    st.markdown("<div style='height:30vh;'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<h1 style='text-align:center;'>What paper are we exploring?</h1>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
        if uploaded_file:
            pdf_bytes = uploaded_file.read()
            with st.status("Thinking...", expanded=True):
                chain = process_document(pdf_bytes)
                summary = chain.invoke({"question": "Summarize this paper in 2 sentences."})["answer"]
            st.session_state.update({
                "chain": chain, "doc_name": uploaded_file.name, "abstract": summary,
                "pdf_b64": base64.b64encode(pdf_bytes).decode("utf-8")
            })
            st.rerun()
else:
    col_pdf, col_qa = st.columns([0.45, 0.55])
    
    with col_pdf:
        st.markdown(f"<p style='color:var(--text-dim); font-size:0.9rem; margin-bottom:10px;'>{st.session_state.doc_name}</p>", unsafe_allow_html=True)
        pdf_url = f"data:application/pdf;base64,{st.session_state.pdf_b64}#page={st.session_state.current_page}"
        st.markdown(f'<iframe src="{pdf_url}" width="100%" height="850vh" class="pdf-frame"></iframe>', unsafe_allow_html=True)

    with col_qa:
        chat_container = st.container(height=800, border=False)
        with chat_container:
            # Welcome Abstract
            st.markdown(f"<div class='ai-msg'><span class='gemini-text'>Summary</span><br><br>{st.session_state.abstract}</div>", unsafe_allow_html=True)
            
            if st.session_state.synthesis:
                st.markdown(f"<div class='ai-msg' style='border: 1px solid #9b72cb;'><span class='gemini-text'>Synthesized Insights</span><br><br>{st.session_state.synthesis}</div>", unsafe_allow_html=True)

            for item in st.session_state.answers:
                st.markdown(f"<div class='user-msg'>{item['question']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='ai-msg'>{item['content']}<br><br><small style='color:var(--gemini-blue);'>Source: Page {item['page']}</small></div>", unsafe_allow_html=True)

        query = st.chat_input("Ask Scriptorium...")
        if query:
            result = st.session_state.chain.invoke({"question": query})
            pg = result["source_documents"][0].metadata.get("page", 0) + 1
            st.session_state.answers.append({"question": query, "content": result["answer"], "page": pg})
            st.session_state.current_page = pg
            st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
