"""Persistence: SQLite, one row per instance holding the Instance as JSON.

JSON rather than normalized tables because the models change weekly right
now - a new field never needs a migration. SQLite still allows querying into
it later (json_extract) for the tracker: "everyone I applied to, no reply".

The database holds real people's data, so it lives in the gitignored .data/.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from schemas import Candidate, Instance, Me

DB_PATH = Path(__file__).resolve().parent / ".data" / "network_ai.db"


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS instances (
        id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, data TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS me (
        id INTEGER PRIMARY KEY CHECK (id = 1), data TEXT NOT NULL)""")
    return conn


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_instance(person: Candidate) -> Instance:
    stamp = now()
    return Instance(id=uuid.uuid4().hex[:12], created_at=stamp, updated_at=stamp, person=person)


def save_instance(instance: Instance, db_path: Path | None = None) -> Instance:
    instance.updated_at = now()
    with _connect(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO instances (id, updated_at, data) VALUES (?, ?, ?)",
                     (instance.id, instance.updated_at, instance.model_dump_json()))
    return instance


def load_instance(instance_id: str, db_path: Path | None = None) -> Instance | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT data FROM instances WHERE id = ?", (instance_id,)).fetchone()
    return Instance.model_validate_json(row[0]) if row else None


def list_instances(db_path: Path | None = None) -> list[Instance]:
    """Most recently touched first."""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT data FROM instances ORDER BY updated_at DESC").fetchall()
    return [Instance.model_validate_json(row[0]) for row in rows]


def delete_instance(instance_id: str, db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM instances WHERE id = ?", (instance_id,))


def load_me(db_path: Path | None = None) -> Me:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT data FROM me WHERE id = 1").fetchone()
    return Me.model_validate_json(row[0]) if row else Me()


def save_me(me: Me, db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO me (id, data) VALUES (1, ?)", (me.model_dump_json(),))
