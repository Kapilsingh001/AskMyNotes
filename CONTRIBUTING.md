# Contributing to AskMyNotes

Thanks for helping out! This is a learning-friendly project — small PRs welcome.

## Local setup
```bash
git clone <this-repo-url>
cd AskMyNotes
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add a free GROQ_API_KEY and a SECRET_KEY
uvicorn main:app --reload
```
Open http://127.0.0.1:8000 and register an account.

> First run downloads the embedding model (~90 MB) and PyTorch — this is normal.

## Ground rules
- **Never commit secrets.** `.env` and `notes.db` are gitignored — keep it that way.
- Match the existing style (clear comments, small functions).
- One focused change per pull request; describe what and why.

## How it fits together
| File | Job |
|------|-----|
| `main.py` | FastAPI app + all endpoints |
| `rag.py` | extract → chunk → embed → similarity search |
| `llm.py` | provider pool (Groq/Gemini/Claude) + all AI generators |
| `models.py` | SQLAlchemy tables (User, Document, Chunk, Flashcard, Review) |
| `auth.py` | password hashing + JWT |
| `scheduler.py` | SM-2 spaced repetition |
| `frontend.html` | the whole UI (vanilla JS) |

## Good first issues
- Add tests (pytest) for `rag.chunk_text` and `scheduler.sm2`.
- Timed mock-exam mode (reuses `/quiz`).
- Flashcard editor (edit/add/delete cards).
- Click-a-citation → open the source page.

See the README's "Ideas to extend" for more.
