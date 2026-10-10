"""Tiny SQLite store for leave requests awaiting manager approval.

The LangGraph checkpointer keeps the paused graph. This table is the
manager's inbox: it lets manager.py list what is pending without having
to scan checkpoints.
"""
import json
import sqlite3
from datetime import datetime, timezone

from .config import REQUESTS_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS leave_requests (
    request_id      TEXT PRIMARY KEY,
    thread_id       TEXT NOT NULL,
    employee_id     TEXT NOT NULL,
    details         TEXT NOT NULL,
    status          TEXT NOT NULL,
    manager_comment TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    decided_at      TEXT
)
"""


def _connect() -> sqlite3.Connection:
    REQUESTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(REQUESTS_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_request(request_id: str, thread_id: str, employee_id: str, details: dict) -> None:
    # INSERT OR IGNORE keeps this safe if a node is ever re-executed.
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO leave_requests "
            "(request_id, thread_id, employee_id, details, status, created_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (request_id, thread_id, employee_id, json.dumps(details), _now()),
        )


def list_pending() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM leave_requests WHERE status = 'pending' ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def get_request(request_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM leave_requests WHERE request_id = ?", (request_id,)
        ).fetchone()
    return dict(row) if row else None


def record_decision(request_id: str, status: str, comment: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE leave_requests SET status = ?, manager_comment = ?, decided_at = ? "
            "WHERE request_id = ?",
            (status, comment, _now(), request_id),
        )