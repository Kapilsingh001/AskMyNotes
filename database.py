"""Database setup — SQLite + SQLAlchemy (same pattern as your JWT project)."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./notes.db"   # a single file; swap for postgresql://... later

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Give each request a DB session, then close it. (FastAPI dependency.)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
