"""SQLite storage backend for Exobrain offline / open-source version."""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone

from app.storage import Document, Snapshot, StorageProtocol

logger = logging.getLogger("exobrain.storage.sqlite")


class SQLiteStorage:
    """Thread-safe SQLite storage using WAL mode."""

    def __init__(self, db_path: str = "app/data/exobrain.db"):
        self.db_path = db_path
        self._lock = threading.Lock()

    async def init(self):
        """Ensure database and tables exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'local',
                    title TEXT NOT NULL DEFAULT 'Untitled Paper',
                    markdown TEXT NOT NULL DEFAULT '',
                    messages TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    markdown TEXT NOT NULL DEFAULT '',
                    messages TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_snaps_doc ON snapshots(document_id, created_at DESC);
            """)
            conn.commit()
            conn.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _row_to_doc(self, row: tuple) -> Document:
        """Convert a row tuple to Document, parsing JSON."""
        id_, user_id, title, markdown, messages_raw, created_at, updated_at = row
        try:
            messages = json.loads(messages_raw)
        except (json.JSONDecodeError, TypeError):
            messages = []
        return Document(
            id=id_, user_id=user_id, title=title, markdown=markdown,
            messages=messages, created_at=created_at, updated_at=updated_at,
        )

    # ── Document CRUD ────────────────────────────────────────────────

    async def create_document(self, user_id: str = "local", title: str = "Untitled Paper") -> Document:
        doc = Document(user_id=user_id, title=title)
        now = self._now()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO documents (id, user_id, title, markdown, messages, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (doc.id, user_id, title, "", "[]", now, now),
            )
            conn.commit()
            conn.close()
        doc.created_at = now
        doc.updated_at = now
        return doc

    async def list_documents(self, user_id: str = "local") -> list[Document]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT id, user_id, title, markdown, messages, created_at, updated_at FROM documents WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
            conn.close()
        return [self._row_to_doc(r) for r in rows]

    async def get_document(self, doc_id: str) -> Document | None:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT id, user_id, title, markdown, messages, created_at, updated_at FROM documents WHERE id=?",
                (doc_id,),
            ).fetchone()
            conn.close()
        if row is None:
            return None
        return self._row_to_doc(row)

    async def update_document(self, doc_id: str, markdown: str, messages: list[dict], title: str | None = None) -> Document | None:
        now = self._now()
        messages_json = json.dumps(messages, ensure_ascii=False)
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            if title is not None:
                conn.execute(
                    "UPDATE documents SET markdown=?, messages=?, title=?, updated_at=? WHERE id=?",
                    (markdown, messages_json, title, now, doc_id),
                )
            else:
                conn.execute(
                    "UPDATE documents SET markdown=?, messages=?, updated_at=? WHERE id=?",
                    (markdown, messages_json, now, doc_id),
                )
            conn.commit()
            # Fetch updated row
            row = conn.execute(
                "SELECT id, user_id, title, markdown, messages, created_at, updated_at FROM documents WHERE id=?",
                (doc_id,),
            ).fetchone()
            conn.close()
        if row is None:
            return None
        return self._row_to_doc(row)

    async def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM snapshots WHERE document_id=?", (doc_id,))
            cursor = conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            conn.close()
        return deleted

    # ── Snapshots ─────────────────────────────────────────────────────

    async def save_snapshot(self, doc_id: str, markdown: str, messages: list[dict]) -> Snapshot:
        snap = Snapshot(document_id=doc_id, markdown=markdown, messages=messages)
        now = self._now()
        messages_json = json.dumps(messages, ensure_ascii=False)
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO snapshots (id, document_id, markdown, messages, created_at) VALUES (?,?,?,?,?)",
                (snap.id, doc_id, markdown, messages_json, now),
            )
            conn.commit()
            conn.close()
        snap.created_at = now
        return snap

    async def list_snapshots(self, doc_id: str) -> list[Snapshot]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT id, document_id, markdown, messages, created_at FROM snapshots WHERE document_id=? ORDER BY created_at DESC",
                (doc_id,),
            ).fetchall()
            conn.close()
        results = []
        for row in rows:
            id_, doc_id_, markdown, messages_raw, created_at = row
            try:
                messages = json.loads(messages_raw)
            except (json.JSONDecodeError, TypeError):
                messages = []
            results.append(Snapshot(id=id_, document_id=doc_id_, markdown=markdown, messages=messages, created_at=created_at))
        return results

    async def restore_snapshot(self, doc_id: str, snapshot_id: str) -> Document | None:
        """Restore a document to a snapshot state. Creates a NEW snapshot of current state first."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            # Get snapshot
            snap_row = conn.execute(
                "SELECT markdown, messages FROM snapshots WHERE id=? AND document_id=?",
                (snapshot_id, doc_id),
            ).fetchone()
            if snap_row is None:
                conn.close()
                return None

            snap_markdown, snap_messages_raw = snap_row
            now = self._now()

            # Save current state as snapshot first (so restore is undoable)
            current = conn.execute("SELECT markdown, messages FROM documents WHERE id=?", (doc_id,)).fetchone()
            if current:
                import uuid
                conn.execute(
                    "INSERT INTO snapshots (id, document_id, markdown, messages, created_at) VALUES (?,?,?,?,?)",
                    (str(uuid.uuid4()), doc_id, current[0], current[1], now),
                )

            # Restore
            conn.execute(
                "UPDATE documents SET markdown=?, messages=?, updated_at=? WHERE id=?",
                (snap_markdown, snap_messages_raw, now, doc_id),
            )
            conn.commit()

            # Fetch restored doc
            row = conn.execute(
                "SELECT id, user_id, title, markdown, messages, created_at, updated_at FROM documents WHERE id=?",
                (doc_id,),
            ).fetchone()
            conn.close()

        return self._row_to_doc(row)
