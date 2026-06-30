"""
Database layer for persisting verification data.
Uses PostgreSQL (via DATABASE_URL) in production, SQLite locally.
"""

import os
import sqlite3
from contextlib import contextmanager

_DATABASE_URL = os.environ.get("DATABASE_URL", "")

if _DATABASE_URL:
    import psycopg2

# ── Connection ────────────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    if _DATABASE_URL:
        url = _DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        conn = psycopg2.connect(url)
    else:
        conn = sqlite3.connect("verifications.db")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS verifications (
                track_id TEXT PRIMARY KEY,
                detection_verified BOOLEAN,
                detection_correct BOOLEAN,
                classification_verified BOOLEAN,
                classification_correct BOOLEAN,
                corrected_name TEXT
            )
        """)


# ── Read ──────────────────────────────────────────────────────────────────────

def load_verifications() -> dict[str, dict]:
    """Return {track_id: {col: value, ...}} for all verified rows."""
    rows = {}
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT track_id, detection_verified, detection_correct, "
                     "classification_verified, classification_correct, corrected_name "
                     "FROM verifications")
        for row in cur.fetchall():
            rows[row[0]] = {
                "detection_verified": row[1],
                "detection_correct": row[2],
                "classification_verified": row[3],
                "classification_correct": row[4],
                "corrected_name": row[5],
            }
    return rows


# ── Write ─────────────────────────────────────────────────────────────────────

def _upsert(track_id: str, **kwargs):
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM verifications WHERE track_id = %s" if _DATABASE_URL
                     else "SELECT 1 FROM verifications WHERE track_id = ?",
                     (track_id,))
        exists = cur.fetchone() is not None

        if exists:
            sets = ", ".join(f"{k} = %s" if _DATABASE_URL else f"{k} = ?"
                             for k in kwargs)
            vals = list(kwargs.values()) + [track_id]
            placeholder = "%s" if _DATABASE_URL else "?"
            cur.execute(f"UPDATE verifications SET {sets} WHERE track_id = {placeholder}", vals)
        else:
            cols = ["track_id"] + list(kwargs.keys())
            placeholder = "%s" if _DATABASE_URL else "?"
            placeholders = ", ".join([placeholder] * len(cols))
            col_str = ", ".join(cols)
            cur.execute(f"INSERT INTO verifications ({col_str}) VALUES ({placeholders})",
                        [track_id] + list(kwargs.values()))


def save_detection(track_id: str, is_correct: bool):
    _upsert(track_id, detection_verified=True, detection_correct=is_correct)


def save_classification(track_id: str, is_correct: bool):
    kwargs = {"classification_verified": True, "classification_correct": is_correct}
    if is_correct:
        kwargs["corrected_name"] = None
    _upsert(track_id, **kwargs)


def save_corrected_name(track_id: str, name: str):
    _upsert(track_id, corrected_name=name)


# Initialize on import
init_db()
