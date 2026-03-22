# ⚗️ ChemInsight AI

**Ask questions about chemistry research papers in plain English.**  
Answers grounded in your document — not the internet, not guesswork.

→ **[Live demo](https://your-app.streamlit.app)** *(update this link after deploying)*

---

## Why I built this

I'm in my final year of BSc (H) Chemistry at Miranda House, and last semester I spent three full weeks doing literature review for my thesis.

The process was: open paper → Ctrl+F → read surrounding paragraphs → repeat. Forty papers. For every single reaction condition I needed to compare. I was doing information retrieval, not chemistry.

So I built something that reads the paper for you. Upload any chemistry PDF and ask it questions the way you'd ask a colleague who just read it — "what catalyst did they use?", "what yield at 60°C?", "explain the mechanism." It finds the exact passage and answers from that. Not from training data. Not from the internet. From your specific document.

I've used it on my own thesis work every week since building it.

---

## What it does

- Upload any chemistry PDF with selectable text
- Ask questions in plain English
- See exactly which page passages the answer came from
- Conversation memory — follow-up questions work naturally
- Detects scanned PDFs and tells you upfront
- Text density chart + indexing analytics per paper
- No signup, no API key needed by visitors — just open and use

---

## How it works

This is a **RAG** (Retrieval-Augmented Generation) system:

```
Your PDF
  ↓ split into 900-character overlapping chunks
  ↓ each chunk → OpenAI embedding (a vector of numbers representing meaning)
  ↓ stored in ChromaDB (local vector database)

When you ask a question:
  ↓ question → embedding
  ↓ MMR search finds 4 most relevant + diverse chunks
  ↓ chunks + question → GPT-4o-mini
  ↓ GPT answers ONLY from those chunks
```

The AI cannot make things up — it only has the retrieved passages to work with.

---

## Stack

| Tool | Role |
|---|---|
| Streamlit | Web interface |
| LangChain | RAG pipeline |
| OpenAI API | GPT-4o-mini (answers) + text-embedding-3-small (vectors) |
| ChromaDB | Local vector database |
| PyPDF | PDF text extraction |
| Plotly | Analytics charts |

---

## Deploy to Streamlit Cloud (free, step by step)

### Step 1 — GitHub
1. Create a repo called `cheminsight-ai` on github.com
2. Upload all project files (app.py, requirements.txt, README.md, .gitignore)
3. Do NOT upload .env

### Step 2 — Streamlit Cloud
1. Go to **share.streamlit.io** → sign in with GitHub
2. Click **New app**
3. Select repo: `ashlesha-sharma/cheminsight-ai`
4. Branch: `main`
5. Main file: `app.py`
6. Click **Advanced settings** → **Secrets**
7. Paste exactly this (with your real key):
   ```toml
   OPENAI_API_KEY = "sk-your-actual-key-here"
   ```
8. Click **Deploy**

Done. Anyone who opens your URL can use the app — no API key needed on their end.

### Step 3 — Update the README
Replace `https://your-app.streamlit.app` at the top with your real Streamlit URL.

---

## Run locally

```bash
git clone https://github.com/ashlesha-sharma/cheminsight-ai
cd cheminsight-ai
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
echo 'OPENAI_API_KEY=sk-your-key' > .env
streamlit run app.py
```

---

## Cost

| Action | Cost |
|---|---|
| Index a 10-page paper | ~$0.001 |
| One question (gpt-4o-mini) | ~$0.003 |
| Full 10-question session | < ₹1 |

Session limit is set to 10 questions per visitor to protect against runaway costs.

---

## What I'd build next

- Multi-paper comparison — "which of these 5 papers reports the best yield for X?"
- Scanned PDF support via OCR
- Reaction scheme extractor — detect and highlight SMILES strings
- Citation export for passages used in answers

---

## About

**Ashlesha Sharma** — BSc (H) Chemistry, Miranda House, University of Delhi  
Building at the intersection of chemistry and AI.

[GitHub](https://github.com/ashlesha-sharma) · [LinkedIn](https://linkedin.com/in/your-profile)
