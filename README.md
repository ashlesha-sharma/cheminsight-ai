# ⚗️ ChemInsight AI

> Ask questions about chemistry research papers in plain English.  
> Answers grounded in your document — not the internet, not guesswork.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat&logo=openai&logoColor=white)

---

## What it does

Upload any chemistry PDF and ask it questions the way you'd ask a colleague who just read it.

- **"What catalyst was used and at what temperature?"**
- **"What yield did they report in the solvent-free experiment?"**
- **"Explain the mechanism they propose in step 3."**
- **"What limitations does the paper mention?"**

The app finds the exact passage in your document and answers from that — not the internet, not training data. Every answer shows which page it came from.

---

## Why it was built

Literature review for a chemistry thesis means opening 40 papers one by one, searching manually for specific reaction conditions, and extracting values by hand. That process takes days and shouldn't.

ChemInsight AI was built to fix exactly that — a tool where you upload the paper and ask it questions instead of reading every line yourself.

Built by **Ashlesha Sharma**, BSc (H) Chemistry, Miranda House, University of Delhi.

---

## How it works

This is a **RAG** (Retrieval-Augmented Generation) system.

```
Your PDF
  │
  ▼
PyPDF extracts all text from every page
  │
  ▼
LangChain splits text into 900-character overlapping chunks
(180-char overlap so no information is lost at boundaries)
  │
  ▼
OpenAI text-embedding-3-small converts each chunk
into a vector — numbers representing its meaning
  │
  ▼
ChromaDB stores all vectors in a local database
  │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When you ask a question:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  ▼
Question → converted to a vector using the same model
  │
  ▼
MMR search finds the 4 most relevant AND diverse chunks
(MMR avoids returning 4 nearly identical passages)
  │
  ▼
Those 4 chunks + your question → GPT-4o-mini
GPT answers ONLY from those chunks — it has nothing
else to work with, so hallucination is not possible
  │
  ▼
Answer shown with page numbers and source passages
```

---

## Features

| Feature | Details |
|---|---|
| PDF upload | Any paper, textbook chapter, or lab report with selectable text |
| RAG pipeline | MMR retrieval — diverse, relevant passages per question |
| Source transparency | Every answer shows which pages it came from, with the exact passage quoted |
| Conversation memory | Follow-up questions work naturally |
| Analytics | Text density bar chart per page, chunk indexing donut, word count |
| Scanned PDF detection | Tells the user immediately if no extractable text is found |
| Session limit | 10 questions per session — protects API costs |
| No login required | Visitors use the app directly — no API key on their end |

---

## Tech stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.11 | Core language |
| Streamlit | 1.35 | Web interface |
| LangChain | 0.2.6 | RAG pipeline orchestration |
| langchain-openai | 0.1.13 | OpenAI integration |
| langchain-community | 0.2.6 | Document loaders and vector store wrappers |
| OpenAI API | — | GPT-4o-mini for answers, text-embedding-3-small for vectors |
| ChromaDB | 0.5.3 | Local vector database |
| PyPDF | 4.2.0 | PDF text extraction |
| Plotly | 5.22.0 | Analytics charts |
| python-dotenv | 1.0.1 | Loads API key from .env in local development |

---

## Project structure

```
cheminsight-ai/
├── app.py              ← Entire application — UI, RAG pipeline, charts
├── requirements.txt    ← All Python dependencies with pinned versions
├── README.md           ← This file
└── .gitignore          ← Prevents API keys and generated folders being committed
```

`app.py` is structured in this order top to bottom:

1. Imports and API key loading
2. Constants — model names, chunk sizes, session limit
3. Page config and full custom CSS
4. Session state initialisation
5. Sidebar — usage meter, how RAG works explanation, about section
6. API key guard — stops the app if no key is found
7. `build_chain()` — processes PDF and builds the RAG chain (cached)
8. `get_sources()` — formats source passages for display
9. `density_chart()` — Plotly bar chart of text per page
10. `donut_chart()` — Plotly donut of chunks indexed
11. Hero section
12. Left column — upload, progress bar, stats, charts, quick questions
13. Right column — chat history, chat input, session limit wall, clear button

---

## Running locally

### Prerequisites
- Python 3.11 or higher
- An OpenAI account with API credits ($5 is enough for months of use on gpt-4o-mini)

### Setup

```bash
# Clone the repository
git clone https://github.com/ashlesha-sharma/cheminsight-ai
cd cheminsight-ai

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
echo 'OPENAI_API_KEY=sk-your-key-here' > .env

# Run
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Getting an OpenAI API key
1. Go to [platform.openai.com](https://platform.openai.com) and create an account
2. Go to **API Keys** → **Create new secret key** → copy it (shown only once)
3. Go to **Billing** → add a card → load $5 in credits
4. Paste the key into your `.env` file as shown above

---

## Deploying to Streamlit Cloud

Streamlit Cloud is free and provides a shareable public URL. The API key lives in their encrypted secrets system — visitors never see it and do not need their own.

### Step 1 — GitHub
Push all four project files to a public GitHub repository. Do not include `.env`.

### Step 2 — Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
2. Click **New app**
3. Select: repository `ashlesha-sharma/cheminsight-ai`, branch `main`, main file `app.py`
4. Click **Advanced settings** → **Secrets**
5. Paste this with your actual key:
   ```toml
   OPENAI_API_KEY = "sk-your-actual-key-here"
   ```
6. Click **Deploy** — live in about 2 minutes

### How the key is loaded
The app checks two sources in order:
1. `st.secrets` — used on Streamlit Cloud
2. `os.getenv` reading `.env` — used locally

The same code works in both environments without any changes.

---

## Cost reference

| Action | Approximate cost |
|---|---|
| Index a 10-page paper | $0.001 |
| One question on gpt-4o-mini | $0.003 |
| Full 10-question session | < $0.03 |
| 100 sessions | ~$3.00 |

The session limit is controlled by the `MAX_QUESTIONS` constant at the top of `app.py`. Change it to any value.

---

## Limitations

- **Scanned PDFs** — if a PDF contains photographed or scanned pages with no selectable text, the app detects this and shows a clear error message.
- **Large papers** — papers over ~80 pages work but indexing takes 30–60 seconds.
- **Figures and reaction schemes** — the app reads text only. Images embedded in PDFs (including reaction diagrams) are not processed.
- **Language** — optimised for English-language papers.

---

## Contributing

Pull requests are welcome.

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Make changes in `app.py`
4. Test locally: `streamlit run app.py`
5. Open a pull request describing what changed and why

---

## About

**Ashlesha Sharma**  
BSc (H) Chemistry, Miranda House, University of Delhi  
[github.com/ashlesha-sharma](https://github.com/ashlesha-sharma)
