"""SQLite persistence for the support desk.

Same database file as :class:`src.storage.InquiryStore`, same rules:
parameterized queries only, idempotent ``CREATE TABLE IF NOT EXISTS`` schema,
driver errors wrapped into :class:`StorageError`. Desk tables reference each
other with real foreign keys (enforced via ``PRAGMA foreign_keys = ON`` in
the shared connection manager). Initializing over an existing Phase 2
database only adds tables — existing data is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.desk.models import (
    AGENT_ROLES,
    Agent,
    AuditEntry,
    Customer,
    InternalNote,
    Tag,
    Ticket,
    TicketEvent,
    TicketMessage,
    new_id,
)
from src.storage import SQLiteConnectionManager, StorageError, utc_now_iso

_DESK_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    external_id TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    email      TEXT NOT NULL UNIQUE,
    role       TEXT NOT NULL CHECK (role IN ('agent', 'admin')),
    active     INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tags (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    id                     TEXT PRIMARY KEY,
    customer_id            TEXT NOT NULL REFERENCES customers(id),
    subject                TEXT NOT NULL,
    status                 TEXT NOT NULL CHECK (status IN ('open', 'pending', 'resolved', 'closed')),
    priority               TEXT NOT NULL CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    assignee_id            TEXT REFERENCES agents(id),
    content_hash           TEXT,
    resolved_at            TEXT,
    ai_summary             TEXT,
    ai_key_points          TEXT,
    ai_category            TEXT,
    ai_urgency             TEXT,
    ai_sentiment           TEXT,
    ai_suggested_priority  TEXT,
    ai_analyzed_at         TEXT,
    ai_provider            TEXT,
    ai_model               TEXT,
    ai_tokens              INTEGER,
    ai_reply               TEXT,
    ai_reply_generated_at  TEXT,
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tickets_status     ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_priority   ON tickets(priority);
CREATE INDEX IF NOT EXISTS idx_tickets_assignee   ON tickets(assignee_id);
CREATE INDEX IF NOT EXISTS idx_tickets_customer   ON tickets(customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_updated    ON tickets(updated_at);
CREATE INDEX IF NOT EXISTS idx_tickets_hash       ON tickets(content_hash);

CREATE TABLE IF NOT EXISTS ticket_messages (
    id           TEXT PRIMARY KEY,
    ticket_id    TEXT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    author_type  TEXT NOT NULL CHECK (author_type IN ('customer', 'agent', 'system')),
    author_name  TEXT NOT NULL,
    body         TEXT NOT NULL,
    content_hash TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_ticket ON ticket_messages(ticket_id);

CREATE TABLE IF NOT EXISTS internal_notes (
    id          TEXT PRIMARY KEY,
    ticket_id   TEXT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    author_id   TEXT NOT NULL,
    author_name TEXT NOT NULL,
    body        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_ticket ON internal_notes(ticket_id);

CREATE TABLE IF NOT EXISTS ticket_tags (
    ticket_id  TEXT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    tag_id     TEXT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (ticket_id, tag_id)
);

CREATE TABLE IF NOT EXISTS ticket_events (
    id         TEXT PRIMARY KEY,
    ticket_id  TEXT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    actor_id   TEXT,
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ticket ON ticket_events(ticket_id);

CREATE TABLE IF NOT EXISTS audit_log (
    id          TEXT PRIMARY KEY,
    actor_id    TEXT,
    action      TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);
"""


def _dump_metadata(metadata: dict) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def _load_metadata(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class DeskStore:
    """Data access layer for customers, agents, tags, tickets and history."""

    def __init__(self, db_path: str | Path) -> None:
        self._connections = SQLiteConnectionManager(db_path)

    def initialize(self) -> None:
        with self._connections.connect() as connection:
            connection.executescript(_DESK_SCHEMA)

    def ping(self) -> bool:
        try:
            with self._connections.connect() as connection:
                connection.execute("SELECT 1").fetchone()
            return True
        except StorageError:
            return False

    # -- customers -----------------------------------------------------------

    def insert_customer(
        self, *, name: str, email: str, external_id: str | None = None
    ) -> tuple[Customer, bool]:
        """Insert a customer; returns ``(customer, created)`` — ``False`` if the
        email already exists (the existing customer is returned)."""
        now = utc_now_iso()
        customer = Customer(
            id=new_id(), name=name, email=email, external_id=external_id,
            created_at=now, updated_at=now,
        )
        with self._connections.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO customers (id, name, email, external_id, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (customer.id, customer.name, customer.email, customer.external_id,
                 customer.created_at, customer.updated_at),
            )
            if cursor.rowcount == 1:
                return customer, True
            row = connection.execute(
                "SELECT * FROM customers WHERE email = ?", (email,)
            ).fetchone()
            if row is None:  # pragma: no cover - defensive
                raise StorageError(f"customer with email {email!r} disappeared mid-insert")
            return self._customer_from_row(row), False

    def get_customer(self, customer_id: str) -> Customer | None:
        with self._connections.connect() as connection:
            row = connection.execute(
                "SELECT * FROM customers WHERE id = ?", (customer_id,)
            ).fetchone()
            return self._customer_from_row(row) if row else None

    def list_customers(self, search: str | None = None, *, limit: int = 50, offset: int = 0) -> list[Customer]:
        query = "SELECT * FROM customers"
        params: list[Any] = []
        if search:
            pattern = f"%{_escape_like(search)}%"
            query += " WHERE name LIKE ? ESCAPE '\\' OR email LIKE ? ESCAPE '\\'"
            params += [pattern, pattern]
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._connections.connect() as connection:
            rows = connection.execute(query, params).fetchall()
            return [self._customer_from_row(row) for row in rows]

    # -- agents ---------------------------------------------------------------

    def insert_agent(
        self, *, name: str, email: str, role: str, active: bool = True
    ) -> tuple[Agent, bool]:
        now = utc_now_iso()
        agent = Agent(
            id=new_id(), name=name, email=email, role=role, active=active,
            created_at=now, updated_at=now,
        )
        with self._connections.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO agents (id, name, email, role, active, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (agent.id, agent.name, agent.email, agent.role, int(agent.active),
                 agent.created_at, agent.updated_at),
            )
            if cursor.rowcount == 1:
                return agent, True
            row = connection.execute(
                "SELECT * FROM agents WHERE email = ?", (email,)
            ).fetchone()
            if row is None:  # pragma: no cover - defensive
                raise StorageError(f"agent with email {email!r} disappeared mid-insert")
            return self._agent_from_row(row), False

    def get_agent(self, agent_id: str) -> Agent | None:
        with self._connections.connect() as connection:
            row = connection.execute(
                "SELECT * FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
            return self._agent_from_row(row) if row else None

    def list_agents(self, *, active_only: bool = False) -> list[Agent]:
        query = "SELECT * FROM agents"
        if active_only:
            query += " WHERE active = 1"
        query += " ORDER BY created_at"
        with self._connections.connect() as connection:
            rows = connection.execute(query).fetchall()
            return [self._agent_from_row(row) for row in rows]

    # -- tags -----------------------------------------------------------------

    def insert_tag(self, name: str) -> tuple[Tag, bool]:
        tag = Tag(id=new_id(), name=name, created_at=utc_now_iso())
        with self._connections.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO tags (id, name, created_at) VALUES (?, ?, ?)",
                (tag.id, tag.name, tag.created_at),
            )
            if cursor.rowcount == 1:
                return tag, True
            row = connection.execute(
                "SELECT * FROM tags WHERE name = ? COLLATE NOCASE", (name,)
            ).fetchone()
            if row is None:  # pragma: no cover - defensive
                raise StorageError(f"tag {name!r} disappeared mid-insert")
            return self._tag_from_row(row), False

    def get_tag(self, tag_id: str) -> Tag | None:
        with self._connections.connect() as connection:
            row = connection.execute("SELECT * FROM tags WHERE id = ?", (tag_id,)).fetchone()
            return self._tag_from_row(row) if row else None

    def list_tags(self) -> list[Tag]:
        with self._connections.connect() as connection:
            rows = connection.execute("SELECT * FROM tags ORDER BY name COLLATE NOCASE").fetchall()
            return [self._tag_from_row(row) for row in rows]

    # -- tickets ---------------------------------------------------------------

    def insert_ticket(self, ticket: Ticket) -> bool:
        with self._connections.connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO tickets (
                    id, customer_id, subject, status, priority, assignee_id, content_hash,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket.id, ticket.customer_id, ticket.subject, ticket.status,
                    ticket.priority, ticket.assignee_id, ticket.content_hash,
                    ticket.created_at, ticket.updated_at,
                ),
            )
            return cursor.rowcount == 1

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        with self._connections.connect() as connection:
            row = connection.execute(
                "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
            ).fetchone()
            return self._ticket_from_row(row) if row else None

    def find_duplicate_ticket(self, content_hash: str) -> Ticket | None:
        """Most recent non-closed ticket with the same content hash, if any."""
        with self._connections.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM tickets
                WHERE content_hash = ? AND status != 'closed'
                ORDER BY created_at DESC LIMIT 1
                """,
                (content_hash,),
            ).fetchone()
            return self._ticket_from_row(row) if row else None

    def update_ticket_fields(self, ticket_id: str, *, fields: dict[str, Any]) -> None:
        """Update an explicit allowlist of ticket columns; bumps ``updated_at``."""
        allowed = {
            "subject", "status", "priority", "assignee_id", "resolved_at",
            "ai_summary", "ai_key_points", "ai_category", "ai_urgency",
            "ai_sentiment", "ai_suggested_priority", "ai_analyzed_at",
            "ai_provider", "ai_model", "ai_tokens", "ai_reply",
            "ai_reply_generated_at",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise StorageError(f"cannot update ticket field(s): {sorted(unknown)}")
        parameters: list[Any] = []
        for name, value in fields.items():
            if name == "ai_key_points":
                value = json.dumps(value, ensure_ascii=False)  # list -> JSON text
            parameters.append(value)
        assignments = ", ".join(f"{name} = ?" for name in fields)
        if assignments:
            assignments += ", "
        assignments += "updated_at = ?"
        parameters += [utc_now_iso(), ticket_id]
        with self._connections.connect() as connection:
            cursor = connection.execute(
                f"UPDATE tickets SET {assignments} WHERE id = ?",  # noqa: S608 - allowlisted columns
                parameters,
            )
            if cursor.rowcount != 1:
                raise StorageError(f"cannot update missing ticket {ticket_id!r}")

    def list_tickets(
        self,
        *,
        statuses: list[str] | None = None,
        priorities: list[str] | None = None,
        assignee_id: str | None = None,
        unassigned: bool = False,
        customer_id: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Ticket]:
        query = (
            "SELECT DISTINCT t.* FROM tickets t"
            " JOIN customers c ON c.id = t.customer_id"
            " LEFT JOIN ticket_messages m ON m.ticket_id = t.id"
        )
        conditions: list[str] = []
        params: list[Any] = []
        if statuses:
            conditions.append(f"t.status IN ({', '.join('?' * len(statuses))})")
            params += statuses
        if priorities:
            conditions.append(f"t.priority IN ({', '.join('?' * len(priorities))})")
            params += priorities
        if unassigned:
            conditions.append("t.assignee_id IS NULL")
        elif assignee_id:
            conditions.append("t.assignee_id = ?")
            params.append(assignee_id)
        if customer_id:
            conditions.append("t.customer_id = ?")
            params.append(customer_id)
        if tag:
            conditions.append(
                "EXISTS (SELECT 1 FROM ticket_tags tt JOIN tags g ON g.id = tt.tag_id"
                " WHERE tt.ticket_id = t.id AND g.name = ? COLLATE NOCASE)"
            )
            params.append(tag)
        if search:
            pattern = f"%{_escape_like(search)}%"
            like = " LIKE ? ESCAPE '\\'"
            conditions.append(
                f"(t.subject{like} OR c.name{like} OR c.email{like} OR m.body{like})"
            )
            params += [pattern] * 4
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY t.updated_at DESC, t.id LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._connections.connect() as connection:
            rows = connection.execute(query, params).fetchall()
            return [self._ticket_from_row(row) for row in rows]

    def count_tickets(self, **filters: Any) -> int:
        tickets = self.list_tickets(limit=10_000, offset=0, **filters)
        return len(tickets)

    # -- messages / notes / tags-on-ticket --------------------------------------

    def insert_message(self, message: TicketMessage) -> bool:
        with self._connections.connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO ticket_messages
                    (id, ticket_id, author_type, author_name, body, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id, message.ticket_id, message.author_type,
                    message.author_name, message.body, message.content_hash,
                    message.created_at,
                ),
            )
            return cursor.rowcount == 1

    def messages_for_ticket(self, ticket_id: str) -> list[TicketMessage]:
        with self._connections.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ticket_messages WHERE ticket_id = ? ORDER BY created_at, id",
                (ticket_id,),
            ).fetchall()
            return [
                TicketMessage(
                    id=row["id"], ticket_id=row["ticket_id"], author_type=row["author_type"],
                    author_name=row["author_name"], body=row["body"],
                    content_hash=row["content_hash"], created_at=row["created_at"],
                )
                for row in rows
            ]

    def insert_note(self, note: InternalNote) -> None:
        with self._connections.connect() as connection:
            connection.execute(
                """
                INSERT INTO internal_notes
                    (id, ticket_id, author_id, author_name, body, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (note.id, note.ticket_id, note.author_id, note.author_name,
                 note.body, note.created_at),
            )

    def notes_for_ticket(self, ticket_id: str) -> list[InternalNote]:
        with self._connections.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM internal_notes WHERE ticket_id = ? ORDER BY created_at, id",
                (ticket_id,),
            ).fetchall()
            return [
                InternalNote(
                    id=row["id"], ticket_id=row["ticket_id"], author_id=row["author_id"],
                    author_name=row["author_name"], body=row["body"], created_at=row["created_at"],
                )
                for row in rows
            ]

    def add_ticket_tag(self, ticket_id: str, tag_id: str) -> bool:
        with self._connections.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO ticket_tags (ticket_id, tag_id, created_at) VALUES (?, ?, ?)",
                (ticket_id, tag_id, utc_now_iso()),
            )
            return cursor.rowcount == 1

    def remove_ticket_tag(self, ticket_id: str, tag_id: str) -> bool:
        with self._connections.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM ticket_tags WHERE ticket_id = ? AND tag_id = ?",
                (ticket_id, tag_id),
            )
            return cursor.rowcount == 1

    def tags_for_ticket(self, ticket_id: str) -> list[Tag]:
        with self._connections.connect() as connection:
            rows = connection.execute(
                """
                SELECT g.* FROM tags g
                JOIN ticket_tags tt ON tt.tag_id = g.id
                WHERE tt.ticket_id = ?
                ORDER BY g.name COLLATE NOCASE
                """,
                (ticket_id,),
            ).fetchall()
            return [self._tag_from_row(row) for row in rows]

    # -- history + audit ---------------------------------------------------------

    def insert_event(
        self, *, ticket_id: str, event_type: str,
        actor_id: str | None, metadata: dict,
    ) -> TicketEvent:
        event = TicketEvent(
            id=new_id(), ticket_id=ticket_id, event_type=event_type,
            actor_id=actor_id, metadata=metadata, created_at=utc_now_iso(),
        )
        with self._connections.connect() as connection:
            connection.execute(
                """
                INSERT INTO ticket_events (id, ticket_id, event_type, actor_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event.id, event.ticket_id, event.event_type, event.actor_id,
                 _dump_metadata(event.metadata), event.created_at),
            )
        return event

    def events_for_ticket(self, ticket_id: str) -> list[TicketEvent]:
        with self._connections.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ticket_events WHERE ticket_id = ? ORDER BY created_at, id",
                (ticket_id,),
            ).fetchall()
            return [
                TicketEvent(
                    id=row["id"], ticket_id=row["ticket_id"], event_type=row["event_type"],
                    actor_id=row["actor_id"], metadata=_load_metadata(row["metadata"]),
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def insert_audit(
        self, *, action: str, entity_type: str, entity_id: str,
        actor_id: str | None, metadata: dict,
    ) -> AuditEntry:
        entry = AuditEntry(
            id=new_id(), action=action, entity_type=entity_type, entity_id=entity_id,
            actor_id=actor_id, metadata=metadata, created_at=utc_now_iso(),
        )
        with self._connections.connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_log (id, actor_id, action, entity_type, entity_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (entry.id, entry.actor_id, entry.action, entry.entity_type,
                 entry.entity_id, _dump_metadata(entry.metadata), entry.created_at),
            )
        return entry

    def list_audit(self, *, limit: int = 100, offset: int = 0) -> list[AuditEntry]:
        with self._connections.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_log ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [
                AuditEntry(
                    id=row["id"], action=row["action"], entity_type=row["entity_type"],
                    entity_id=row["entity_id"], actor_id=row["actor_id"],
                    metadata=_load_metadata(row["metadata"]), created_at=row["created_at"],
                )
                for row in rows
            ]

    # -- row mappers ---------------------------------------------------------------

    @staticmethod
    def _customer_from_row(row) -> Customer:
        return Customer(
            id=row["id"], name=row["name"], email=row["email"],
            external_id=row["external_id"], created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _agent_from_row(row) -> Agent:
        role = row["role"] if row["role"] in AGENT_ROLES else "agent"
        return Agent(
            id=row["id"], name=row["name"], email=row["email"], role=role,
            active=bool(row["active"]), created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _tag_from_row(row) -> Tag:
        return Tag(id=row["id"], name=row["name"], created_at=row["created_at"])

    @staticmethod
    def _ticket_from_row(row) -> Ticket:
        try:
            key_points = json.loads(row["ai_key_points"] or "[]")
        except (TypeError, ValueError):
            key_points = []
        if not isinstance(key_points, list):
            key_points = []
        return Ticket(
            id=row["id"], customer_id=row["customer_id"], subject=row["subject"],
            status=row["status"], priority=row["priority"],
            assignee_id=row["assignee_id"], content_hash=row["content_hash"],
            resolved_at=row["resolved_at"], ai_summary=row["ai_summary"],
            ai_key_points=[str(item) for item in key_points],
            ai_category=row["ai_category"], ai_urgency=row["ai_urgency"],
            ai_sentiment=row["ai_sentiment"], ai_suggested_priority=row["ai_suggested_priority"],
            ai_analyzed_at=row["ai_analyzed_at"], ai_provider=row["ai_provider"],
            ai_model=row["ai_model"], ai_tokens=row["ai_tokens"],
            ai_reply=row["ai_reply"], ai_reply_generated_at=row["ai_reply_generated_at"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )


def _escape_like(term: str) -> str:
    """Escape LIKE wildcards so user search input matches literally."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
