"""Postgres access. One sync connection pool, shared by every component.

Sync on purpose: inference is a blocking GPU/CPU workload, and FastAPI runs
`def` endpoints in a threadpool, so an async DB driver would buy nothing and
cost a duplicated code path. The same DSN points at the local Postgres
container or at an Aurora PostgreSQL writer endpoint.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from fleet.config import settings

log = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=8,
            kwargs={"options": "-c search_path=fleet,public"},
            open=True,
        )
        _pool.wait(timeout=30)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def conn() -> Iterator[Any]:
    with pool().connection() as c:
        c.row_factory = dict_row
        yield c


def fetchall(sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
    with conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def fetchone(sql: str, params: Sequence[Any] | None = None) -> dict[str, Any] | None:
    with conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def execute(sql: str, params: Sequence[Any] | None = None) -> int:
    with conn() as c, c.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def apply_schema(path: Path | None = None) -> None:
    """Idempotent DDL apply. Safe to call on every container start."""
    ddl_path = path or (Path(__file__).resolve().parents[2] / "sql" / "001_schema.sql")
    ddl = ddl_path.read_text()
    with pool().connection() as c:
        c.execute(ddl)
    log.info("schema applied from %s", ddl_path)


def wait_for_db(timeout_s: float = 60.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            fetchone("SELECT 1 AS ok")
            return
        except Exception as exc:  # noqa: BLE001 - retry loop on purpose
            last = exc
            close_pool()
            time.sleep(1.0)
    raise RuntimeError(f"database not reachable at {settings.database_url}: {last}")
