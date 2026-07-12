"""Exobrain storage abstraction — SQLite for offline, Postgres for Railway."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# ── Models (shared between backends) ──────────────────────────────────

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class Document:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "local"  # "local" for offline, UUID for online
    title: str = "Untitled Paper"
    markdown: str = ""
    messages: list[dict] = field(default_factory=list)  # [{role, content}, ...]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "markdown": self.markdown,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: dict) -> "Document":
        return cls(
            id=row["id"],
            user_id=row.get("user_id", "local"),
            title=row.get("title", "Untitled Paper"),
            markdown=row.get("markdown", ""),
            messages=row.get("messages", []),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
        )


@dataclass
class Snapshot:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    markdown: str = ""
    messages: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "markdown": self.markdown,
            "messages": self.messages,
            "created_at": self.created_at,
        }


# ── Protocol ──────────────────────────────────────────────────────────

@runtime_checkable
class StorageProtocol(Protocol):
    """Unified storage interface. Implementations: SQLite, Postgres."""

    async def create_document(self, user_id: str, title: str = "Untitled Paper") -> Document: ...
    async def list_documents(self, user_id: str) -> list[Document]: ...
    async def get_document(self, doc_id: str) -> Document | None: ...
    async def update_document(self, doc_id: str, markdown: str, messages: list[dict], title: str | None = None) -> Document | None: ...
    async def delete_document(self, doc_id: str) -> bool: ...
    async def save_snapshot(self, doc_id: str, markdown: str, messages: list[dict]) -> Snapshot: ...
    async def list_snapshots(self, doc_id: str) -> list[Snapshot]: ...
    async def restore_snapshot(self, doc_id: str, snapshot_id: str) -> Document | None: ...


# ── Factory ────────────────────────────────────────────────────────────

import os

_storage: StorageProtocol | None = None


async def get_storage() -> StorageProtocol:
    global _storage
    if _storage is not None:
        return _storage

    backend = os.getenv("EXOBRAIN_STORAGE", "sqlite")
    if backend == "postgres":
        from app.storage.postgres_storage import PostgresStorage
        _storage = PostgresStorage()
    else:
        from app.storage.sqlite_storage import SQLiteStorage
        db_path = os.getenv("EXOBRAIN_SQLITE_PATH", "app/data/exobrain.db")
        _storage = SQLiteStorage(db_path=db_path)

    await _storage.init()
    return _storage
