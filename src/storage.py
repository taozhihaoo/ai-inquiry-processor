"""SQLite persistence layer — a small, explicit data access layer on stdlib sqlite3.

Design rules:
- Parameterized queries only, never string-concatenated SQL.
- Schema is created idempotently (``CREATE TABLE IF NOT EXISTS``) on init.
- All operations wrap ``sqlite3.Error`` into :class:`StorageError` so callers
  (service / API) can convert failures into safe responses without leaking
  driver internals.
- One connection per operation for file databases (safe across threads);
  ``:memory:`` databases keep a single shared connection instead.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

MEMORY_DB = ":memory:"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS processing_runs (
    run_id      TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    model       TEXT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    total       INTEGER,
    succeeded   INTEGER,
    failed      INTEGER
);

CREATE TABLE IF NOT EXISTS inquiries (
    id             TEXT PRIMARY KEY,
    customer_name  TEXT NOT NULL,
    message        TEXT NOT NULL,
    status         TEXT NOT NULL,
    summary        TEXT,
    category       TEXT,
    priority       TEXT,
    error_message  TEXT,
    run_id         TEXT,
    provider       TEXT NOT NULL,
    model          TEXT,
    input_tokens   INTEGER,
    output_tokens  INTEGER,
    total_tokens   INTEGER,
    latency_ms     REAL,
    attempts       INTEGER,
    estimated_cost REAL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_inquiries_run_id ON inquiries(run_id);
CREATE INDEX IF NOT EXISTS idx_inquiries_status ON inquiries(status);
"""


class StorageError(Exception):
    """Raised when a persistence operation fails."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class SQLiteConnectionManager:
    """Owns connection handling for one SQLite database path.

    File databases use one connection per operation (safe across threads);
    ``:memory:`` databases keep a single shared connection. Foreign-key
    enforcement is enabled on every connection, and driver errors are wrapped
    into :class:`StorageError`.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._memory = self._path == MEMORY_DB
        self._shared: sqlite3.Connection | None = None
        if not self._memory:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory:
            if self._shared is None:
                self._shared = sqlite3.connect(MEMORY_DB, check_same_thread=False)
                self._shared.row_factory = sqlite3.Row
            self._shared.execute("PRAGMA foreign_keys = ON")
            yield self._shared
            self._shared.commit()
            return
        try:
            connection = sqlite3.connect(self._path)
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open database {self._path!r}: {exc}") from exc
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise StorageError(f"database operation failed: {exc}") from exc
        finally:
            connection.close()


@dataclass
class InquiryRecord:
    """One inquiry as stored in SQLite; ``None`` fields mean not available (yet)."""

    id: str
    customer_name: str
    message: str
    status: str
    provider: str
    summary: str | None = None
    category: str | None = None
    priority: str | None = None
    error_message: str | None = None
    run_id: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    attempts: int | None = None
    estimated_cost: float | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def pending(
        cls,
        *,
        id: str,
        customer_name: str,
        message: str,
        provider: str,
        model: str | None,
        run_id: str | None = None,
    ) -> InquiryRecord:
        now = utc_now_iso()
        return cls(
            id=id,
            customer_name=customer_name,
            message=message,
            status="pending",
            provider=provider,
            model=model,
            run_id=run_id,
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> InquiryRecord:
        return cls(**dict(row))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_success(self) -> bool:
        return self.status == "success"


@dataclass
class ProcessingRun:
    run_id: str
    provider: str
    model: str | None
    started_at: str
    finished_at: str | None = None
    total: int | None = None
    succeeded: int | None = None
    failed: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class InquiryStore:
    """SQLite-backed store for inquiries and processing runs."""

    def __init__(self, db_path: str | Path) -> None:
        self._connections = SQLiteConnectionManager(db_path)

    def initialize(self) -> None:
        """Create the schema if it does not exist yet (safe to call repeatedly)."""
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    # -- inquiries ---------------------------------------------------------

    def insert_inquiry(self, record: InquiryRecord) -> bool:
        """Insert a new inquiry row; return ``False`` if the id already exists."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO inquiries (
                    id, customer_name, message, status, provider, model, run_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id, record.customer_name, record.message, record.status,
                    record.provider, record.model, record.run_id,
                    record.created_at, record.updated_at,
                ),
            )
            return cursor.rowcount == 1

    def get_inquiry(self, inquiry_id: str) -> InquiryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM inquiries WHERE id = ?", (inquiry_id,)
            ).fetchone()
            return InquiryRecord.from_row(row) if row is not None else None

    def update_success(
        self,
        inquiry_id: str,
        *,
        summary: str,
        category: str,
        priority: str,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: float | None = None,
        attempts: int | None = None,
        estimated_cost: float | None = None,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE inquiries SET
                    status = 'success', summary = ?, category = ?, priority = ?,
                    model = COALESCE(?, model),
                    input_tokens = ?, output_tokens = ?, total_tokens = ?,
                    latency_ms = ?, attempts = ?, estimated_cost = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    summary, category, priority,
                    model,
                    input_tokens, output_tokens, total_tokens,
                    latency_ms, attempts, estimated_cost,
                    utc_now_iso(), inquiry_id,
                ),
            )
            if cursor.rowcount != 1:
                raise StorageError(f"cannot update missing inquiry {inquiry_id!r}")

    def update_error(
        self,
        inquiry_id: str,
        *,
        error_message: str,
        attempts: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE inquiries SET
                    status = 'error', error_message = ?,
                    attempts = COALESCE(?, attempts), latency_ms = COALESCE(?, latency_ms),
                    updated_at = ?
                WHERE id = ?
                """,
                (error_message, attempts, latency_ms, utc_now_iso(), inquiry_id),
            )
            if cursor.rowcount != 1:
                raise StorageError(f"cannot update missing inquiry {inquiry_id!r}")

    def inquiries_for_run(self, run_id: str) -> list[InquiryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM inquiries WHERE run_id = ? ORDER BY created_at, id", (run_id,)
            ).fetchall()
            return [InquiryRecord.from_row(row) for row in rows]

    def count_inquiries(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM inquiries").fetchone()
            return int(row["n"])

    # -- processing runs ---------------------------------------------------

    def create_run(self, run_id: str, *, provider: str, model: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO processing_runs (run_id, provider, model, started_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, provider, model, utc_now_iso()),
            )

    def finish_run(self, run_id: str, *, total: int, succeeded: int, failed: int) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE processing_runs
                SET finished_at = ?, total = ?, succeeded = ?, failed = ?
                WHERE run_id = ?
                """,
                (utc_now_iso(), total, succeeded, failed, run_id),
            )
            if cursor.rowcount != 1:
                raise StorageError(f"cannot finish missing run {run_id!r}")

    def get_run(self, run_id: str) -> ProcessingRun | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM processing_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            return ProcessingRun(
                run_id=row["run_id"],
                provider=row["provider"],
                model=row["model"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                total=row["total"],
                succeeded=row["succeeded"],
                failed=row["failed"],
            )

    def ping(self) -> bool:
        """Cheap liveness probe used by the health endpoint."""
        try:
            with self._connect() as connection:
                connection.execute("SELECT 1").fetchone()
            return True
        except StorageError:
            return False

    # -- connection handling -------------------------------------------------

    def _connect(self):
        return self._connections.connect()
