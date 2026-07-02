"""FastAPI app that wires everything together.

Auth:
  POST /auth/register / /auth/login -> get a JWT; /auth/me -> who am I
Everything below is scoped to the logged-in user (send: Authorization: Bearer <token>).

Core RAG:
  POST /upload | /upload-text  -> add notes (PDF/DOCX/TXT or pasted text)
  POST /ask | /chat            -> ask / chat over your notes (cited, memory in chat)
  GET  /documents | /subjects  -> your library
Study features (need an AI key):
  POST /summary /quiz /flashcards/generate /testme /grade
  GET  /flashcards/due /flashcards /flashcards/export
  POST /flashcards/{id}/review -> SM-2 + logs a review
  GET  /stats                  -> streak, due, mastered, accuracy (the dashboard)
"""

from dotenv import load_dotenv
load_dotenv()   # load API keys / SECRET_KEY from .env (before anything else)

import json
import secrets
from typing import Optional
from datetime import datetime, date, timedelta

from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

import models
import rag
import llm
import auth
import scheduler
from database import engine, get_db, SessionLocal

models.Base.metadata.create_all(bind=engine)


def _migrate():
    """Add columns introduced after a DB was first created (SQLite create_all won't)."""
    from sqlalchemy import text
    with engine.connect() as conn:
        fc = {r[1] for r in conn.execute(text("PRAGMA table_info(flashcards)"))}
        for col, typ in {"difficulty": "VARCHAR", "topic": "VARCHAR", "est_time": "INTEGER",
                         "source_page": "INTEGER", "hint": "TEXT", "explanation": "TEXT"}.items():
            if col not in fc:
                conn.execute(text(f"ALTER TABLE flashcards ADD COLUMN {col} {typ}"))
        docs = {r[1] for r in conn.execute(text("PRAGMA table_info(documents)"))}
        if "user_id" not in docs:
            conn.execute(text("ALTER TABLE documents ADD COLUMN user_id INTEGER"))
        if "share_token" not in docs:
            conn.execute(text("ALTER TABLE documents ADD COLUMN share_token VARCHAR"))
        conn.commit()


def _assign_orphans():
    """Give any pre-auth documents to a 'demo' account so existing data isn't lost."""
    db = SessionLocal()
    try:
        orphans = db.query(models.Document).filter(models.Document.user_id.is_(None)).all()
        if orphans:
            demo = db.query(models.User).filter_by(email="demo@askmynotes.local").first()
            if not demo:
                demo = models.User(email="demo@askmynotes.local",
                                   password_hash=auth.hash_password("demo1234"))
                db.add(demo)
                db.commit()
                db.refresh(demo)
            for d in orphans:
                d.user_id = demo.id
            db.commit()
    finally:
        db.close()


_migrate()
_assign_orphans()

app = FastAPI(title="AskMyNotes")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------- request bodies ----------
class AuthRequest(BaseModel):
    email: str
    password: str


class AskRequest(BaseModel):
    question: str
    subject: Optional[str] = None
    document_id: Optional[int] = None


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []          # [{"role": "user"|"assistant", "content": "..."}]
    subject: Optional[str] = None
    document_id: Optional[int] = None


class TextUploadRequest(BaseModel):
    title: str
    text: str
    subject: str = "General"


class ScopeRequest(BaseModel):
    document_id: Optional[int] = None
    subject: Optional[str] = None
    num: int = 5


class DocRequest(BaseModel):
    document_id: int
    num: int = 10


class ReviewRequest(BaseModel):
    quality: int   # 0-5, how well you recalled the card


class GradeRequest(BaseModel):
    question: str
    answer: str
    subject: Optional[str] = None
    document_id: Optional[int] = None


# ---------- auth dependency ----------
def current_user(authorization: Optional[str] = Header(None),
                 db: Session = Depends(get_db)) -> models.User:
    """Resolve the Bearer token to a User, or 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    token = authorization.split(" ", 1)[1]
    try:
        payload = auth.decode_token(token)
        user = db.query(models.User).get(int(payload["sub"]))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please log in again.")
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


# ---------- helpers (all user-scoped) ----------
def _scoped_chunks(db, user, document_id=None, subject=None, limit=None) -> list:
    q = db.query(models.Chunk).join(models.Document).filter(models.Document.user_id == user.id)
    if document_id is not None:
        q = q.filter(models.Chunk.document_id == document_id)
    elif subject:
        q = q.filter(models.Document.subject == subject)
    if limit:
        q = q.limit(limit)
    return q.all()


def _owned_doc(db, user, document_id) -> models.Document:
    doc = db.query(models.Document).filter_by(id=document_id, user_id=user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


def _require_key():
    if not llm.has_key():
        raise HTTPException(
            status_code=400,
            detail="This feature needs an API key (GROQ_API_KEY, GEMINI_API_KEY or ANTHROPIC_API_KEY). Add it to a .env file to enable AI study tools.",
        )


def _ai(fn, *args, **kwargs):
    """Run an AI generator, turning provider failures into a clean 503 (not a raw 500)."""
    try:
        return fn(*args, **kwargs)
    except llm.AIError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------- static / UI ----------
@app.get("/")
def home():
    return FileResponse("frontend.html", headers={"Cache-Control": "no-store"})


@app.get("/manifest.json")
def manifest():
    return FileResponse("manifest.json", media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return FileResponse("sw.js", media_type="application/javascript",
                        headers={"Cache-Control": "no-store"})


# ---------- auth ----------
@app.post("/auth/register")
def register(req: AuthRequest, db: Session = Depends(get_db)):
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    if db.query(models.User).filter_by(email=email).first():
        raise HTTPException(status_code=400, detail="That email is already registered.")
    user = models.User(email=email, password_hash=auth.hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"token": auth.make_token(user.id), "email": user.email}


@app.post("/auth/login")
def login(req: AuthRequest, db: Session = Depends(get_db)):
    email = req.email.strip().lower()
    user = db.query(models.User).filter_by(email=email).first()
    if not user or not auth.verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Wrong email or password.")
    return {"token": auth.make_token(user.id), "email": user.email}


@app.get("/auth/me")
def me(user: models.User = Depends(current_user)):
    return {"id": user.id, "email": user.email}


# ---------- ingestion ----------
@app.post("/upload")
async def upload(file: UploadFile = File(...), subject: str = Form("General"),
                 user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Please upload a PDF, DOCX or TXT file.")

    file_bytes = await file.read()
    try:
        pages = rag.extract_pages_any(file.filename, file_bytes)   # OCR fallback inside for PDFs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not any(text.strip() for _, text in pages):
        raise HTTPException(status_code=400, detail="Could not read any text from that file.")

    doc = models.Document(user_id=user.id, filename=file.filename,
                          subject=subject.strip() or "General")
    db.add(doc)
    db.commit()
    db.refresh(doc)

    total_chunks = 0
    for page_no, text in pages:
        if not text.strip():
            continue
        chunks = rag.chunk_text(text)
        if not chunks:
            continue
        embeddings = rag.embed_texts(chunks)
        for content, vector in zip(chunks, embeddings):
            db.add(models.Chunk(document_id=doc.id, content=content,
                                embedding=json.dumps(vector), page=page_no))
            total_chunks += 1
    db.commit()
    return {"document_id": doc.id, "filename": doc.filename,
            "subject": doc.subject, "chunks_created": total_chunks}


@app.post("/upload-text")
def upload_text(req: TextUploadRequest, user: models.User = Depends(current_user),
                db: Session = Depends(get_db)):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Paste some text first.")
    doc = models.Document(user_id=user.id, filename=req.title.strip() or "Pasted note",
                          subject=req.subject.strip() or "General")
    db.add(doc)
    db.commit()
    db.refresh(doc)

    chunks = rag.chunk_text(req.text)
    embeddings = rag.embed_texts(chunks)
    for content, vector in zip(chunks, embeddings):
        db.add(models.Chunk(document_id=doc.id, content=content,
                            embedding=json.dumps(vector), page=None))
    db.commit()
    return {"document_id": doc.id, "filename": doc.filename,
            "subject": doc.subject, "chunks_created": len(chunks)}


# ---------- library ----------
@app.get("/documents")
def list_documents(user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    docs = (db.query(models.Document).filter_by(user_id=user.id)
              .order_by(models.Document.created_at.desc()).all())
    return [{"id": d.id, "filename": d.filename, "subject": d.subject,
             "chunks": len(d.chunks), "flashcards": len(d.flashcards),
             "has_summary": bool(d.summary), "share_token": d.share_token} for d in docs]


@app.get("/subjects")
def list_subjects(user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    rows = (db.query(models.Document.subject, func.count(models.Document.id))
              .filter(models.Document.user_id == user.id)
              .group_by(models.Document.subject).all())
    return [{"subject": s, "documents": n} for s, n in rows]


# ---------- sharing (public deck links) ----------
@app.post("/documents/{doc_id}/share")
def share_document(doc_id: int, user: models.User = Depends(current_user),
                   db: Session = Depends(get_db)):
    """Turn on a public link for this deck's flashcards; returns the share token."""
    doc = _owned_doc(db, user, doc_id)
    if not doc.share_token:
        doc.share_token = secrets.token_urlsafe(8)
        db.commit()
    return {"share_token": doc.share_token, "path": f"/s/{doc.share_token}"}


@app.delete("/documents/{doc_id}/share")
def unshare_document(doc_id: int, user: models.User = Depends(current_user),
                     db: Session = Depends(get_db)):
    doc = _owned_doc(db, user, doc_id)
    doc.share_token = None
    db.commit()
    return {"ok": True}


@app.get("/api/shared/{token}")
def shared_deck(token: str, db: Session = Depends(get_db)):
    """PUBLIC (no auth): the flashcards behind a share link, read-only."""
    doc = db.query(models.Document).filter_by(share_token=token).first()
    if not doc:
        raise HTTPException(status_code=404, detail="This shared deck was not found or was unshared.")
    cards = db.query(models.Flashcard).filter_by(document_id=doc.id).all()
    return {"title": doc.filename, "subject": doc.subject,
            "cards": [{"front": c.front, "back": c.back,
                       "difficulty": c.difficulty, "topic": c.topic} for c in cards]}


@app.get("/s/{token}")
def shared_viewer(token: str):
    """PUBLIC: the little viewer page for a shared deck."""
    return FileResponse("shared.html", headers={"Cache-Control": "no-store"})


# ---------- ask / chat ----------
@app.post("/ask")
def ask(req: AskRequest, user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    chunks = _scoped_chunks(db, user, req.document_id, req.subject)
    if not chunks:
        raise HTTPException(status_code=400, detail="No notes in that scope yet. Upload a file first.")
    top = rag.search(req.question, chunks)
    labeled, sources = rag.label_chunks(top)
    return {"answer": llm.generate_answer(req.question, labeled), "sources": sources}


@app.post("/chat")
def chat(req: ChatRequest, user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    chunks = _scoped_chunks(db, user, req.document_id, req.subject)
    if not chunks:
        raise HTTPException(status_code=400, detail="No notes in that scope yet. Upload a file first.")
    top = rag.search(req.question, chunks)
    labeled, sources = rag.label_chunks(top)
    return {"answer": llm.generate_chat(req.question, labeled, req.history), "sources": sources}


# ---------- cheat-sheet ----------
@app.post("/summary")
def summary(req: DocRequest, user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    _require_key()
    doc = _owned_doc(db, user, req.document_id)
    if doc.summary:
        return {"document_id": doc.id, "summary": doc.summary, "cached": True}
    labeled, _ = rag.label_chunks(_scoped_chunks(db, user, document_id=doc.id, limit=12))
    text = _ai(llm.generate_summary, labeled)
    doc.summary = text
    db.commit()
    return {"document_id": doc.id, "summary": text, "cached": False}


# ---------- quiz ----------
@app.post("/quiz")
def quiz(req: ScopeRequest, user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    _require_key()
    chunks = _scoped_chunks(db, user, req.document_id, req.subject, limit=12)
    if not chunks:
        raise HTTPException(status_code=400, detail="No notes in that scope yet.")
    labeled, _ = rag.label_chunks(chunks)
    return {"questions": _ai(llm.generate_quiz, labeled, num=req.num)}


# ---------- flashcards ----------
@app.post("/flashcards/generate")
def flashcards_generate(req: DocRequest, user: models.User = Depends(current_user),
                        db: Session = Depends(get_db)):
    _require_key()
    doc = _owned_doc(db, user, req.document_id)
    labeled, _ = rag.label_chunks(_scoped_chunks(db, user, document_id=doc.id, limit=12))
    cards = _ai(llm.generate_flashcards, labeled, num=req.num)  # only replace old cards if this succeeds

    db.query(models.Flashcard).filter_by(document_id=doc.id).delete()

    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    created = 0
    for c in cards:
        q = c.get("question") or c.get("front")
        a = c.get("answer") or c.get("back")
        if not q or not a:
            continue
        db.add(models.Flashcard(
            document_id=doc.id, front=q, back=a,
            difficulty=(c.get("difficulty") or None), topic=(c.get("topic") or None),
            est_time=_int(c.get("est_time")), source_page=_int(c.get("source_page")),
            hint=(c.get("hint") or None), explanation=(c.get("explanation") or None)))
        created += 1
    db.commit()
    return {"created": created, "document_id": doc.id}


@app.get("/flashcards/due")
def flashcards_due(subject: Optional[str] = None, difficulty: Optional[str] = None,
                   user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    q = (db.query(models.Flashcard).join(models.Document)
           .filter(models.Document.user_id == user.id,
                   models.Flashcard.due_date <= datetime.utcnow()))
    if subject:
        q = q.filter(models.Document.subject == subject)
    if difficulty:
        q = q.filter(models.Flashcard.difficulty == difficulty)
    return [{"id": c.id, "front": c.front, "back": c.back,
             "difficulty": c.difficulty, "topic": c.topic,
             "est_time": c.est_time, "source_page": c.source_page,
             "hint": c.hint, "explanation": c.explanation,
             "document": c.document.filename} for c in q.all()]


@app.get("/flashcards")
def flashcards_list(document_id: int, user: models.User = Depends(current_user),
                    db: Session = Depends(get_db)):
    _owned_doc(db, user, document_id)
    cards = db.query(models.Flashcard).filter_by(document_id=document_id).all()
    return [{"id": c.id, "front": c.front, "back": c.back,
             "difficulty": c.difficulty, "topic": c.topic,
             "est_time": c.est_time, "source_page": c.source_page,
             "hint": c.hint, "explanation": c.explanation,
             "due_date": c.due_date.isoformat() if c.due_date else None,
             "repetitions": c.repetitions} for c in cards]


@app.get("/flashcards/export")
def flashcards_export(subject: Optional[str] = None, document_id: Optional[int] = None,
                      user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    """Export cards as CSV (Anki-importable: Front, Back, Topic, Difficulty)."""
    import csv, io
    q = db.query(models.Flashcard).join(models.Document).filter(models.Document.user_id == user.id)
    if document_id is not None:
        q = q.filter(models.Flashcard.document_id == document_id)
    elif subject:
        q = q.filter(models.Document.subject == subject)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Front", "Back", "Topic", "Difficulty"])
    for c in q.all():
        w.writerow([c.front, c.back, c.topic or "", c.difficulty or ""])
    return PlainTextResponse(buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=askmynotes_flashcards.csv"})


@app.post("/flashcards/{card_id}/review")
def flashcards_review(card_id: int, req: ReviewRequest,
                      user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    card = (db.query(models.Flashcard).join(models.Document)
              .filter(models.Flashcard.id == card_id, models.Document.user_id == user.id).first())
    if not card:
        raise HTTPException(status_code=404, detail="Card not found.")
    reps, ease, interval, due = scheduler.sm2(req.quality, card.repetitions, card.ease, card.interval)
    card.repetitions, card.ease, card.interval = reps, ease, interval
    card.due_date, card.last_reviewed = due, datetime.utcnow()
    db.add(models.Review(user_id=user.id, card_id=card.id, quality=req.quality))  # for streaks/stats
    db.commit()
    return {"id": card.id, "next_due": due.isoformat(), "interval_days": interval, "repetitions": reps}


# ---------- 'test me' ----------
@app.post("/testme")
def testme(req: ScopeRequest, user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    _require_key()
    chunk = db.query(models.Chunk).join(models.Document).filter(models.Document.user_id == user.id)
    if req.document_id is not None:
        chunk = chunk.filter(models.Chunk.document_id == req.document_id)
    elif req.subject:
        chunk = chunk.filter(models.Document.subject == req.subject)
    chunk = chunk.order_by(func.random()).first()
    if not chunk:
        raise HTTPException(status_code=400, detail="No notes in that scope yet.")
    labeled, _ = rag.label_chunks([chunk])
    question = llm.generate_answer(
        "Write ONE short open-ended exam question whose answer is contained in these "
        "notes. Return only the question, nothing else.", labeled)
    return {"question": question.strip()}


@app.post("/grade")
def grade(req: GradeRequest, user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    _require_key()
    chunks = _scoped_chunks(db, user, req.document_id, req.subject)
    if not chunks:
        raise HTTPException(status_code=400, detail="No notes in that scope yet.")
    top = rag.search(req.question, chunks)
    labeled, _ = rag.label_chunks(top)
    return _ai(llm.grade_answer, req.question, req.answer, labeled)


# ---------- progress dashboard ----------
@app.get("/stats")
def stats(user: models.User = Depends(current_user), db: Session = Depends(get_db)):
    docs = db.query(models.Document).filter_by(user_id=user.id).count()
    cards_q = db.query(models.Flashcard).join(models.Document).filter(models.Document.user_id == user.id)
    total_cards = cards_q.count()
    due_today = cards_q.filter(models.Flashcard.due_date <= datetime.utcnow()).count()
    mastered = cards_q.filter(models.Flashcard.repetitions >= 3).count()

    reviews = db.query(models.Review).filter_by(user_id=user.id).all()
    today = date.today()
    review_dates = {r.reviewed_at.date() for r in reviews}
    reviews_today = sum(1 for r in reviews if r.reviewed_at.date() == today)

    # streak = consecutive days (ending today, or yesterday if nothing yet today) with a review
    streak, day = 0, today
    if today not in review_dates and (today - timedelta(days=1)) in review_dates:
        day = today - timedelta(days=1)
    while day in review_dates:
        streak += 1
        day -= timedelta(days=1)

    # 7-day accuracy (quality >= 3 counts as "recalled")
    wk_start = today - timedelta(days=6)
    recent = [r for r in reviews if r.reviewed_at.date() >= wk_start]
    accuracy = round(100 * sum(1 for r in recent if r.quality >= 3) / len(recent)) if recent else None

    last7 = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        last7.append({"day": d.isoformat(), "count": sum(1 for r in reviews if r.reviewed_at.date() == d)})

    return {"email": user.email, "documents": docs, "total_cards": total_cards,
            "due_today": due_today, "mastered": mastered, "reviews_today": reviews_today,
            "total_reviews": len(reviews), "streak": streak, "accuracy_7d": accuracy,
            "last7": last7}
