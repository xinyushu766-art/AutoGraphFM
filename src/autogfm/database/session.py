"""Engine and session lifecycle helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from autogfm.database.settings import DatabaseSettings, get_database_settings


def create_database_engine(
    settings: DatabaseSettings | None = None,
) -> Engine:
    """Create a SQLAlchemy engine without opening a connection immediately."""
    resolved = settings or get_database_settings()
    engine_kwargs: dict[str, object] = {
        "echo": resolved.database_echo,
        "pool_pre_ping": True,
    }

    if resolved.database_url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in resolved.database_url:
            engine_kwargs["poolclass"] = StaticPool

    return create_engine(
        resolved.database_url,
        **engine_kwargs,
    )


def create_session_factory(
    engine: Engine,
) -> sessionmaker[Session]:
    """Create a typed session factory."""
    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


@contextmanager
def session_scope(
    factory: sessionmaker[Session] | None = None,
) -> Iterator[Session]:
    """Commit on success and roll back on failure."""
    if factory is None:
        engine = create_database_engine()
        factory = create_session_factory(engine)

    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def database_healthcheck(engine: Engine) -> bool:
    """Return true when the database accepts a trivial query."""
    try:
        with engine.connect() as connection:
            connection.execute(select(1))
        return True
    except Exception:
        return False
