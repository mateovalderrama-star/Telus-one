"""SQLite-backed append-only AuditResult store."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from caz_sentinel.types import AuditResult


class AuditStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS audits (
                    request_id TEXT PRIMARY KEY,
                    ts_ns INTEGER NOT NULL,
                    payload TEXT NOT NULL
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def append(self, a: AuditResult) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO audits VALUES (?, ?, ?)",
                      (a.request_id, a.timestamp_ns, json.dumps(a.to_dict())))

    def get(self, request_id: str) -> AuditResult | None:
        with self._conn() as c:
            row = c.execute("SELECT payload FROM audits WHERE request_id=?", (request_id,)).fetchone()
        return AuditResult.from_dict(json.loads(row["payload"])) if row else None

    def list(self, limit: int = 100) -> list[AuditResult]:
        with self._conn() as c:
            rows = c.execute("SELECT payload FROM audits ORDER BY ts_ns DESC LIMIT ?",
                             (limit,)).fetchall()
        return [AuditResult.from_dict(json.loads(r["payload"])) for r in rows]
