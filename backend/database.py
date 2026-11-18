"""SQLAlchemy database engine and session management."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    """Declarative base for ORM models."""


def create_engine_and_session(db_url: str | None = None):
    """Create SQLAlchemy engine and session factory for given database URL."""
    url = db_url or get_settings().db_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, future=True)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return engine, session_factory


_settings = get_settings()
engine, SessionLocal = create_engine_and_session(_settings.db_url)


def init_db(target_engine=None) -> None:
    """Create database tables if they do not yet exist."""
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=target_engine or engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
