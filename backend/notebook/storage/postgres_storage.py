"""PostgreSQL storage backend — for Railway production deployment."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone

from app.storage import Document, Snapshot

logger = logging.getLogger("exobrain.storage.postgres")

# Lazy import — production only
_pool = None


def _get_pool():
    """Lazy-init psycopg2 connection pool."""
    global _pool
    if _pool is None:
        import psycopg2
        import psycopg2.pool
        db_url = os.getenv("DATABASE_URL", "")
        if not db_url:
            raise RuntimeError("DATABASE_URL must be set for Postgres storage")
        _pool = psycopg2.pool.ThreadedConnectionPool(1, 10, db_url)
    return _pool


class PostgresStorage:
    """Postgres backend — same interface as SQLiteStorage."""

    async def init(self):
        pool = _get_pool()
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS exobrain_documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'local',
                    title TEXT NOT NULL DEFAULT 'Untitled Paper',
                    markdown TEXT NOT NULL DEFAULT '',
                    messages JSONB NOT NULL DEFAULT '[]',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS exobrain_snapshots (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES exobrain_documents(id) ON DELETE CASCADE,
                    markdown TEXT NOT NULL DEFAULT '',
                    messages JSONB NOT NULL DEFAULT '[]',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_exo_docs_user ON exobrain_documents(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_exo_snaps_doc ON exobrain_snapshots(document_id, created_at DESC);
            """)
            conn.commit()
            cur.close()
        finally:
            pool.putconn(conn)

    def _row_to_doc(self, row: tuple) -> Document:
        id_, user_id, title, markdown, messages, created_at, updated_at = row
        if isinstance(messages, str):
            messages = json.loads(messages)
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        if isinstance(updated_at, datetime):
            updated_at = updated_at.isoformat()
        return Document(id=id_, user_id=user_id, title=title, markdown=markdown,
                        messages=messages, created_at=created_at, updated_at=updated_at)

    # ── Document CRUD ────────────────────────────────────────────────

    async def create_document(self, user_id: str = "local", title: str = "Untitled Paper") -> Document:
        doc = Document(user_id=user_id, title=title)
        pool = _get_pool()
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO exobrain_documents (id, user_id, title, markdown, messages) VALUES (%s,%s,%s,%s,%s)",
                (doc.id, user_id, title, "", "[]"),
            )
            conn.commit()
            cur.close()
        finally:
            pool.putconn(conn)
        return doc

    async def list_documents(self, user_id: str = "local") -> list[Document]:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, user_id, title, markdown, messages, created_at, updated_at FROM exobrain_documents WHERE user_id=%s ORDER BY updated_at DESC",
                (user_id,),
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            pool.putconn(conn)
        return [self._row_to_doc(r) for r in rows]

    async def get_document(self, doc_id: str) -> Document | None:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, user_id, title, markdown, messages, created_at, updated_at FROM exobrain_documents WHERE id=%s",
                (doc_id,),
            )
            row = cur.fetchone()
            cur.close()
        finally:
            pool.putconn(conn)
        if row is None:
            return None
        return self._row_to_doc(row)

    async def update_document(self, doc_id: str, markdown: str, messages: list[dict], title: str | None = None) -> Document | None:
        messages_json = json.dumps(messages, ensure_ascii=False)
        pool = _get_pool()
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            if title is not None:
                cur.execute(
                    "UPDATE exobrain_documents SET markdown=%s, messages=%s, title=%s, updated_at=NOW() WHERE id=%s",
                    (markdown, messages_json, title, doc_id),
                )
            else:
                cur.execute(
                    "UPDATE exobrain_documents SET markdown=%s, messages=%s, updated_at=NOW() WHERE id=%s",
                    (markdown, messages_json, doc_id),
                )
            conn.commit()

            cur.execute(
                "SELECT id, user_id, title, markdown, messages, created_at, updated_at FROM exobrain_documents WHERE id=%s",
                (doc_id,),
            )
            row = cur.fetchone()
            cur.close()
        finally:
            pool.putconn(conn)
        if row is None:
            return None
        return self._row_to_doc(row)

    async def delete_document(self, doc_id: str) -> bool:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM exobrain_snapshots WHERE document_id=%s", (doc_id,))
            cur.execute("DELETE FROM exobrain_documents WHERE id=%s", (doc_id,))
            deleted = cur.rowcount > 0
            conn.commit()
            cur.close()
        finally:
            pool.putconn(conn)
        return deleted

    # ── Snapshots ─────────────────────────────────────────────────────

    async def save_snapshot(self, doc_id: str, markdown: str, messages: list[dict]) -> Snapshot:
        snap = Snapshot(document_id=doc_id, markdown=markdown, messages=messages)
        messages_json = json.dumps(messages, ensure_ascii=False)
        pool = _get_pool()
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO exobrain_snapshots (id, document_id, markdown, messages) VALUES (%s,%s,%s,%s)",
                (snap.id, doc_id, markdown, messages_json),
            )
            conn.commit()
            cur.close()
        finally:
            pool.putconn(conn)
        return snap

    async def list_snapshots(self, doc_id: str) -> list[Snapshot]:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, document_id, markdown, messages, created_at FROM exobrain_snapshots WHERE document_id=%s ORDER BY created_at DESC",
                (doc_id,),
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            pool.putconn(conn)
        results = []
        for row in rows:
            id_, doc_id_, markdown, messages_raw, created_at = row
            if isinstance(messages_raw, str):
                messages = json.loads(messages_raw)
            else:
                messages = messages_raw
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            results.append(Snapshot(id=id_, document_id=doc_id_, markdown=markdown, messages=messages, created_at=created_at))
        return results

    async def restore_snapshot(self, doc_id: str, snapshot_id: str) -> Document | None:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            cur = conn.cursor()

            # Get snapshot
            cur.execute(
                "SELECT markdown, messages FROM exobrain_snapshots WHERE id=%s AND document_id=%s",
                (snapshot_id, doc_id),
            )
            snap_row = cur.fetchone()
            if snap_row is None:
                cur.close()
                return None

            snap_markdown, snap_messages = snap_row

            # Save current state as snapshot first
            cur.execute(
                "SELECT markdown, messages FROM exobrain_documents WHERE id=%s", (doc_id,),
            )
            current = cur.fetchone()
            if current:
                cur.execute(
                    "INSERT INTO exobrain_snapshots (id, document_id, markdown, messages) VALUES (%s,%s,%s,%s)",
                    (str(uuid.uuid4()), doc_id, current[0],
                     json.dumps(current[1]) if isinstance(current[1], (list, dict)) else str(current[1])),
                )

            cur.execute(
                "UPDATE exobrain_documents SET markdown=%s, messages=%s, updated_at=NOW() WHERE id=%s",
                (snap_markdown, json.dumps(snap_messages) if isinstance(snap_messages, (list, dict)) else str(snap_messages), doc_id),
            )
            conn.commit()

            cur.execute(
                "SELECT id, user_id, title, markdown, messages, created_at, updated_at FROM exobrain_documents WHERE id=%s",
                (doc_id,),
            )
            row = cur.fetchone()
            cur.close()
        finally:
            pool.putconn(conn)
        return self._row_to_doc(row)
