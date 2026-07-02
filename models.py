"""Database tables.

Document ──< Chunk        (a document is split into many embedded chunks)
Document ──< Flashcard    (a document can have many spaced-repetition cards)

Each Chunk stores its embedding (as JSON) and the page it came from (for citations).
Each Flashcard carries SM-2 spaced-repetition state (when to review it next).
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # owner
    filename = Column(String, nullable=False)
    subject = Column(String, nullable=False, default="General", index=True)  # folder / tag
    summary = Column(Text, nullable=True)                # cached cheat-sheet (generated on demand)
    share_token = Column(String, nullable=True, index=True)  # set when the deck is publicly shared
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    flashcards = relationship("Flashcard", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)        # the raw text of this chunk
    embedding = Column(Text, nullable=False)      # the vector, stored as a JSON string
    page = Column(Integer, nullable=True)         # page it came from (for citations)

    document = relationship("Document", back_populates="chunks")


class Flashcard(Base):
    __tablename__ = "flashcards"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    front = Column(Text, nullable=False)          # the question / prompt side
    back = Column(Text, nullable=False)           # the answer side (may use markdown bullets)
    difficulty = Column(String, nullable=True)    # Easy / Medium / Hard
    topic = Column(String, nullable=True)         # short topic label, e.g. "Normalization"
    est_time = Column(Integer, nullable=True)     # estimated recall time in seconds (10-60)
    source_page = Column(Integer, nullable=True)  # page the concept came from
    hint = Column(Text, nullable=True)            # optional nudge shown before the answer
    explanation = Column(Text, nullable=True)     # short "why" shown with the answer

    # --- SM-2 spaced-repetition state ---
    ease = Column(Float, default=2.5)             # how "easy" the card is (>=1.3)
    interval = Column(Integer, default=0)         # days until next review
    repetitions = Column(Integer, default=0)      # consecutive correct recalls
    due_date = Column(DateTime, default=datetime.utcnow)   # when it's next due (now = due immediately)
    last_reviewed = Column(DateTime, nullable=True)

    document = relationship("Document", back_populates="flashcards")


class Review(Base):
    """One row per flashcard review — the history behind streaks & stats."""
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    card_id = Column(Integer, ForeignKey("flashcards.id"), nullable=True)
    quality = Column(Integer, nullable=False)            # 0-5 self-rating
    reviewed_at = Column(DateTime, default=datetime.utcnow, index=True)
