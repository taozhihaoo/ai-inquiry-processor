"""Domain models for the support desk (dataclasses, matching the project style).

Enum vocabularies are plain tuples of strings — the single source of truth for
storage CHECK constraints, API validation and service-level checks, mirroring
``src.models``. AI-derived fields are stored as *advisory* values: they are
classifications / suggestions, never treated as facts decided by the system.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field

TICKET_STATUSES = ("open", "pending", "resolved", "closed")
TICKET_PRIORITIES = ("low", "normal", "high", "urgent")
URGENCY_LEVELS = ("low", "normal", "high", "urgent")
SENTIMENTS = ("positive", "neutral", "negative")
AGENT_ROLES = ("agent", "admin")
AUTHOR_TYPES = ("customer", "agent", "system")

OPEN_STATUSES = ("open", "pending")

TICKET_EVENT_TYPES = (
    "ticket_created", "status_changed", "priority_changed", "assigned",
    "unassigned", "tag_added", "tag_removed", "note_added", "message_added",
    "ai_analyzed", "ai_reply_generated",
)

AUDIT_ACTIONS = (
    "ticket_created", "status_changed", "priority_changed", "assigned",
    "unassigned", "tag_added", "tag_removed", "note_added", "message_added",
    "ai_analyzed", "ai_reply_generated", "customer_created", "agent_created",
    "tag_created",
)


class DeskValidationError(ValueError):
    """Raised when a desk-domain value violates its vocabulary or shape."""


def new_id() -> str:
    """Opaque 32-char hex identifier (uuid4), unrelated to content hashes."""
    return uuid.uuid4().hex


def validate_choice(value: str, vocabulary: tuple[str, ...], field_name: str) -> str:
    if value not in vocabulary:
        raise DeskValidationError(
            f"invalid {field_name} {value!r}; expected one of {list(vocabulary)}"
        )
    return value


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_email(email: str) -> str:
    if not _EMAIL_RE.match(email):
        raise DeskValidationError(f"invalid email address: {email!r}")
    return email


def normalize_text(text: str) -> str:
    """Collapse whitespace and lowercase — input to duplicate-detection hashes."""
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass
class Customer:
    id: str
    name: str
    email: str
    created_at: str
    updated_at: str
    external_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Agent:
    id: str
    name: str
    email: str
    role: str
    active: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        data = asdict(self)
        return data

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass
class Tag:
    id: str
    name: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Ticket:
    """A support ticket; ``ai_*`` fields are advisory AI output, nullable forever."""

    id: str
    customer_id: str
    subject: str
    status: str
    priority: str
    assignee_id: str | None
    created_at: str
    updated_at: str
    resolved_at: str | None = None
    content_hash: str | None = None
    ai_summary: str | None = None
    ai_key_points: list[str] = field(default_factory=list)
    ai_category: str | None = None
    ai_urgency: str | None = None
    ai_sentiment: str | None = None
    ai_suggested_priority: str | None = None
    ai_analyzed_at: str | None = None
    ai_provider: str | None = None
    ai_model: str | None = None
    ai_tokens: int | None = None
    ai_reply: str | None = None
    ai_reply_generated_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TicketMessage:
    id: str
    ticket_id: str
    author_type: str
    author_name: str
    body: str
    created_at: str
    content_hash: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InternalNote:
    id: str
    ticket_id: str
    author_id: str
    author_name: str
    body: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TicketEvent:
    id: str
    ticket_id: str
    event_type: str
    actor_id: str | None
    metadata: dict
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditEntry:
    id: str
    action: str
    entity_type: str
    entity_id: str
    actor_id: str | None
    metadata: dict
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)
