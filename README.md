# 🧠 AskMyNotes — an AI Study Assistant (RAG)

Upload your own notes (PDF) and ask questions — answered **only from your material**,
using a Retrieval-Augmented Generation (RAG) pipeline with Claude.

This is a **reference project to study, then rebuild yourself.**

## How it works (RAG in 2 phases)
```
INGEST:  PDF -> extract text -> split into chunks -> embed each chunk -> store vectors
QUERY:   question -> embed -> find closest chunks -> send to Claude -> grounded answer
```

## The pieces
| File          | Job                                                        |
|---------------|------------------------------------------------------------|
| `config.py`   | All settings (chunk size, top-k, model names)              |
| `database.py` | SQLite + SQLAlchemy connection                             |
| `models.py`   | Tables: Document has many Chunks (chunk stores its vector) |
| `rag.py`      | The engine: extract, chunk, embed, similarity search       |
| `llm.py`      | Sends retrieved chunks to Claude, returns the answer       |
| `main.py`     | FastAPI app: /upload, /ask, serves the frontend            |
| `frontend.html` | The web UI                                               |

## Tech choices (and why)
- **Embeddings run locally** via `sentence-transformers` (free, no API key). Anthropic
  has no embeddings endpoint, so RAG apps pair Claude (generation) with a separate
  embedding model.
- **Vector search is plain numpy** (cosine similarity). Fine at learning scale; a real
  app would use pgvector / a vector DB.
- **Generation uses Claude** (`claude-opus-4-8`). Set `CLAUDE_MODEL` in `config.py`
  to `claude-haiku-4-5` to make answers much cheaper while learning.

## Setup (once)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
> First run downloads the embedding model (~90 MB) and installs PyTorch — the install
> is large and needs internet. This is normal.

## Add your Claude key (for AI answers)
1. Get a key at https://console.anthropic.com  (Settings → API Keys)
2. Copy `.env.example` to `.env` and paste your key.
> Retrieval works WITHOUT a key — you'll just see the retrieved chunks instead of an
> AI answer until you add one.

## Run
```
venv\Scripts\activate
uvicorn main:app --reload
```
Then open **http://127.0.0.1:8000/** — upload a PDF, ask a question.

## Ideas to extend (great resume additions)
- Add JWT auth (reuse your Student Portal code) so each user has private notes.
- Stream the answer token-by-token.
- Swap numpy search for **pgvector** in PostgreSQL.
- Add citations (which page each chunk came from).
- Dockerize and deploy.

## Deploy it online (so others can use it)
The project is deploy-ready. **Before deploying, set two things in your host's env vars:**
`SECRET_KEY` (a long random string) and one AI key (`GROQ_API_KEY` recommended — free).

**Option A — Render / Railway / Fly (uses the `Procfile`):**
1. Push this repo to GitHub (the `.gitignore` keeps `.env` and `notes.db` out).
2. Create a new **Web Service** from the repo.
3. Build command: `pip install -r requirements.txt` · Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Add env vars `SECRET_KEY` and `GROQ_API_KEY`. Deploy.

**Option B — Docker (uses the `Dockerfile`):**
```
docker build -t askmynotes .
docker run -p 8000:8000 -e SECRET_KEY=... -e GROQ_API_KEY=... askmynotes
```

> Notes: the first request downloads the embedding model (~90 MB), so cold starts are slow —
> use a plan with ≥1 GB RAM. `notes.db` (SQLite) lives on the container's disk; for permanent
> multi-user data, attach a disk/volume or move to PostgreSQL (swap `DATABASE_URL` in `database.py`).

## Sharing & other features
- **Accounts:** each user has private notes (JWT login). Data is isolated per user.
- **Share a deck:** Library → *Share* generates a public `/s/<token>` link — anyone can study those
  flashcards read-only, no login.
- **Providers:** the app auto-picks Groq → Gemini → Claude by whichever key is in `.env`.
