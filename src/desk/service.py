"""Support desk application service — the single business entry for desk operations.

Layering: API/CLI -> DeskService -> DeskStore (+ LLMClient via desk.ai).
Every mutating call receives the authenticated :class:`AgentContext` and the
service enforces the RBAC rules, the ticket workflow, duplicate detection,
history events and the audit log. Endpoints contain no business logic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.desk import ai as desk_ai
from src.desk.auth import AgentContext, require_admin
from src.desk.models import (
    AUDIT_ACTIONS,
    TICKET_EVENT_TYPES,
    TICKET_PRIORITIES,
    TICKET_STATUSES,
    Agent,
    AuditEntry,
    Customer,
    DeskValidationError,
    InternalNote,
    Tag,
    Ticket,
    TicketEvent,
    TicketMessage,
    new_id,
    normalize_text,
    validate_choice,
    validate_email,
)
from src.desk.storage import DeskStore
from src.desk.workflow import InvalidTransitionError, validate_transition
from src.storage import utc_now_iso

MAX_SEARCH_LENGTH = 200
DUPLICATE_HASH_PREFIX = "ticket-msg-v1"


class DeskError(Exception):
    """Base class for desk business errors (mapped to safe API responses)."""

    status_code = 400
    code = "desk_error"

    def __init__(self, message: str, *, metadata: dict | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


class DeskNotFound(DeskError):
    status_code = 404
    code = "not_found"


class DeskConflict(DeskError):
    status_code = 409
    code = "conflict"


class DeskValidation(DeskError):
    status_code = 400
    code = "validation_error"


class DeskForbidden(DeskError):
    status_code = 403
    code = "forbidden"


@dataclass(frozen=True)
class TicketDetail:
    ticket: Ticket
    customer: Customer | None
    assignee: Agent | None
    tags: list[Tag]
    messages: list[TicketMessage]

    def to_dict(self) -> dict:
        return {
            "ticket": self.ticket.to_dict(),
            "customer": self.customer.to_dict() if self.customer else None,
            "assignee": self.assignee.to_dict() if self.assignee else None,
            "tags": [tag.to_dict() for tag in self.tags],
            "messages": [message.to_dict() for message in self.messages],
        }


class DeskService:
    """Orchestrates desk operations; owns workflow, permissions, history and audit."""

    def __init__(
        self,
        *,
        store: DeskStore,
        llm_client,
        provider: str = "mock",
        model: str | None = None,
    ) -> None:
        self._store = store
        self._llm_client = llm_client
        self._provider = provider
        self._model = model

    @property
    def store(self) -> DeskStore:
        return self._store

    # -- permissions -------------------------------------------------------------

    @staticmethod
    def _can_modify(ticket: Ticket, actor: AgentContext) -> bool:
        """Admins act on any ticket; agents act on unassigned or own tickets."""
        if actor.is_admin:
            return True
        return ticket.assignee_id is None or ticket.assignee_id == actor.agent_id

    def _require_modifiable(self, ticket: Ticket, actor: AgentContext) -> None:
        if not self._can_modify(ticket, actor):
            raise DeskForbidden("ticket is assigned to another agent")

    @staticmethod
    def _validated(value: str, vocabulary: tuple[str, ...], field_name: str) -> str:
        try:
            return validate_choice(value, vocabulary, field_name)
        except DeskValidationError as exc:
            raise DeskValidation(str(exc)) from None

    @staticmethod
    def _validated_email(email: str) -> str:
        try:
            return validate_email(email)
        except DeskValidationError as exc:
            raise DeskValidation(str(exc)) from None

    # -- customers -----------------------------------------------------------------

    def create_customer(
        self, actor: AgentContext, *, name: str, email: str, external_id: str | None = None
    ) -> tuple[Customer, bool]:
        self._validated_email(email)
        customer, created = self._store.insert_customer(
            name=name.strip(), email=email.strip().lower(), external_id=external_id
        )
        if created:
            self._audit(actor, "customer_created", "customer", customer.id,
                        {"email": customer.email})
        return customer, created

    def get_customer(self, actor: AgentContext, customer_id: str) -> Customer:
        customer = self._store.get_customer(customer_id)
        if customer is None:
            raise DeskNotFound(f"customer {customer_id} not found")
        return customer

    def list_customers(self, actor: AgentContext, *, search: str | None = None) -> list[Customer]:
        return self._store.list_customers(self._clean_search(search))

    def customer_tickets(self, actor: AgentContext, customer_id: str) -> list[Ticket]:
        self.get_customer(actor, customer_id)
        return self._store.list_tickets(customer_id=customer_id)

    # -- tickets --------------------------------------------------------------------

    def create_ticket(
        self,
        actor: AgentContext,
        *,
        customer_id: str,
        subject: str,
        message: str,
        priority: str = "normal",
    ) -> Ticket:
        customer = self.get_customer(actor, customer_id)
        self._validated(priority, TICKET_PRIORITIES, "ticket priority")
        content_hash = self._content_hash(customer_id, message)

        duplicate = self._store.find_duplicate_ticket(content_hash)
        if duplicate is not None:
            raise DeskConflict(
                "an open ticket with the same content already exists",
                metadata={
                    "existing_ticket_id": duplicate.id,
                    "existing_status": duplicate.status,
                },
            )

        now = utc_now_iso()
        ticket = Ticket(
            id=new_id(), customer_id=customer.id, subject=subject.strip(),
            status="open", priority=priority, assignee_id=None,
            content_hash=content_hash, created_at=now, updated_at=now,
        )
        if not self._store.insert_ticket(ticket):  # pragma: no cover - uuid4 collision
            raise DeskConflict("ticket id collision, please retry")
        self._store.insert_message(
            TicketMessage(
                id=new_id(), ticket_id=ticket.id, author_type="customer",
                author_name=customer.name, body=message.strip(),
                content_hash=content_hash, created_at=now,
            )
        )
        self._event(ticket.id, actor, "ticket_created",
                    {"subject": ticket.subject, "priority": ticket.priority})
        self._audit(actor, "ticket_created", "ticket", ticket.id,
                    {"customer_id": customer.id, "priority": ticket.priority})
        return ticket

    def get_ticket(self, actor: AgentContext, ticket_id: str) -> Ticket:
        ticket = self._store.get_ticket(ticket_id)
        if ticket is None:
            raise DeskNotFound(f"ticket {ticket_id} not found")
        return ticket

    def ticket_detail(self, actor: AgentContext, ticket_id: str) -> TicketDetail:
        ticket = self.get_ticket(actor, ticket_id)
        return TicketDetail(
            ticket=ticket,
            customer=self._store.get_customer(ticket.customer_id),
            assignee=self._store.get_agent(ticket.assignee_id) if ticket.assignee_id else None,
            tags=self._store.tags_for_ticket(ticket.id),
            messages=self._store.messages_for_ticket(ticket.id),
        )

    def list_tickets(
        self,
        actor: AgentContext,
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
        for status in statuses or []:
            validate_choice(status, TICKET_STATUSES, "ticket status")
        for priority in priorities or []:
            validate_choice(priority, TICKET_PRIORITIES, "ticket priority")
        return self._store.list_tickets(
            statuses=statuses, priorities=priorities, assignee_id=assignee_id,
            unassigned=unassigned, customer_id=customer_id, tag=tag,
            search=self._clean_search(search),
            limit=min(max(limit, 1), 200), offset=max(offset, 0),
        )

    def update_ticket(
        self,
        actor: AgentContext,
        ticket_id: str,
        *,
        status: str | None = None,
        priority: str | None = None,
        subject: str | None = None,
    ) -> Ticket:
        ticket = self.get_ticket(actor, ticket_id)
        self._require_modifiable(ticket, actor)

        fields: dict = {}
        events: list[tuple[str, dict]] = []

        if status is not None and status != ticket.status:
            try:
                validate_transition(ticket.status, status)
            except InvalidTransitionError as exc:
                raise DeskConflict(str(exc)) from None
            fields["status"] = status
            events.append(("status_changed", {"from": ticket.status, "to": status}))
            if status == "resolved":
                fields["resolved_at"] = utc_now_iso()
            elif ticket.status == "resolved" and status == "open":
                fields["resolved_at"] = None  # reopen clears the resolution timestamp

        if priority is not None and priority != ticket.priority:
            self._validated(priority, TICKET_PRIORITIES, "ticket priority")
            fields["priority"] = priority
            events.append(("priority_changed", {"from": ticket.priority, "to": priority}))

        if subject is not None and subject.strip() and subject.strip() != ticket.subject:
            fields["subject"] = subject.strip()

        if fields:
            self._store.update_ticket_fields(ticket_id, fields=fields)
            for event_type, metadata in events:
                self._event(ticket_id, actor, event_type, metadata)
                self._audit(actor, event_type, "ticket", ticket_id, metadata)
        return self.get_ticket(actor, ticket_id)

    def assign_ticket(self, actor: AgentContext, ticket_id: str, assignee_id: str) -> Ticket:
        ticket = self.get_ticket(actor, ticket_id)
        self._require_modifiable(ticket, actor)
        assignee = self._store.get_agent(assignee_id)
        if assignee is None:
            raise DeskNotFound(f"agent {assignee_id} not found")
        if not assignee.active:
            raise DeskValidation("cannot assign to a deactivated agent")
        if ticket.assignee_id == assignee_id:
            return ticket  # no-op
        self._store.update_ticket_fields(ticket_id, fields={"assignee_id": assignee_id})
        self._event(ticket_id, actor, "assigned", {"assignee_id": assignee_id})
        self._audit(actor, "assigned", "ticket", ticket_id, {"assignee_id": assignee_id})
        return self.get_ticket(actor, ticket_id)

    def unassign_ticket(self, actor: AgentContext, ticket_id: str) -> Ticket:
        ticket = self.get_ticket(actor, ticket_id)
        self._require_modifiable(ticket, actor)
        if ticket.assignee_id is None:
            return ticket  # no-op
        previous = ticket.assignee_id
        self._store.update_ticket_fields(ticket_id, fields={"assignee_id": None})
        self._event(ticket_id, actor, "unassigned", {"previous_assignee_id": previous})
        self._audit(actor, "unassigned", "ticket", ticket_id, {"previous_assignee_id": previous})
        return self.get_ticket(actor, ticket_id)

    def agent_tickets(self, actor: AgentContext, agent_id: str) -> list[Ticket]:
        if self._store.get_agent(agent_id) is None:
            raise DeskNotFound(f"agent {agent_id} not found")
        return self._store.list_tickets(assignee_id=agent_id)

    # -- messages / notes / tags ------------------------------------------------------

    def add_customer_message(
        self, actor: AgentContext, ticket_id: str, *, body: str
    ) -> TicketMessage:
        ticket = self.get_ticket(actor, ticket_id)
        customer = self._store.get_customer(ticket.customer_id)
        now = utc_now_iso()
        message = TicketMessage(
            id=new_id(), ticket_id=ticket_id, author_type="customer",
            author_name=customer.name if customer else "customer",
            body=body.strip(), content_hash=self._content_hash(ticket.customer_id, body),
            created_at=now,
        )
        self._store.insert_message(message)
        self._store.update_ticket_fields(ticket_id, fields={})  # bump updated_at
        self._event(ticket_id, actor, "message_added", {"message_id": message.id})
        self._audit(actor, "message_added", "ticket", ticket_id, {"message_id": message.id})
        return message

    def add_note(self, actor: AgentContext, ticket_id: str, *, body: str) -> InternalNote:
        self.get_ticket(actor, ticket_id)
        note = InternalNote(
            id=new_id(), ticket_id=ticket_id, author_id=actor.agent_id,
            author_name=actor.name, body=body.strip(), created_at=utc_now_iso(),
        )
        self._store.insert_note(note)
        self._event(ticket_id, actor, "note_added", {"note_id": note.id})
        self._audit(actor, "note_added", "ticket", ticket_id, {"note_id": note.id})
        return note

    def list_notes(self, actor: AgentContext, ticket_id: str) -> list[InternalNote]:
        self.get_ticket(actor, ticket_id)
        return self._store.notes_for_ticket(ticket_id)

    def create_tag(self, actor: AgentContext, *, name: str) -> Tag:
        require_admin(actor)
        cleaned = name.strip()
        if not cleaned:
            raise DeskValidation("tag name must not be empty")
        tag, created = self._store.insert_tag(cleaned)
        if created:
            self._audit(actor, "tag_created", "tag", tag.id, {"name": tag.name})
        return tag

    def list_tags(self, actor: AgentContext) -> list[Tag]:
        return self._store.list_tags()

    def add_tag(self, actor: AgentContext, ticket_id: str, *, tag_id: str) -> Tag:
        ticket = self.get_ticket(actor, ticket_id)
        self._require_modifiable(ticket, actor)
        tag = self._store.get_tag(tag_id)
        if tag is None:
            raise DeskNotFound(f"tag {tag_id} not found")
        if self._store.add_ticket_tag(ticket_id, tag_id):
            self._event(ticket_id, actor, "tag_added", {"tag": tag.name})
            self._audit(actor, "tag_added", "ticket", ticket_id, {"tag": tag.name})
        return tag

    def remove_tag(self, actor: AgentContext, ticket_id: str, *, tag_id: str) -> None:
        ticket = self.get_ticket(actor, ticket_id)
        self._require_modifiable(ticket, actor)
        tag = self._store.get_tag(tag_id)
        if tag is None:
            raise DeskNotFound(f"tag {tag_id} not found")
        if self._store.remove_ticket_tag(ticket_id, tag_id):
            self._event(ticket_id, actor, "tag_removed", {"tag": tag.name})
            self._audit(actor, "tag_removed", "ticket", ticket_id, {"tag": tag.name})

    def ticket_tags(self, actor: AgentContext, ticket_id: str) -> list[Tag]:
        self.get_ticket(actor, ticket_id)
        return self._store.tags_for_ticket(ticket_id)

    # -- agents ------------------------------------------------------------------------

    def create_agent(
        self, actor: AgentContext, *, name: str, email: str, role: str
    ) -> tuple[Agent, bool]:
        require_admin(actor)
        self._validated_email(email)
        self._validated(role, ("agent", "admin"), "agent role")
        agent, created = self._store.insert_agent(
            name=name.strip(), email=email.strip().lower(), role=role
        )
        if created:
            self._audit(actor, "agent_created", "agent", agent.id,
                        {"email": agent.email, "role": agent.role})
        return agent, created

    def list_agents(self, actor: AgentContext) -> list[Agent]:
        return self._store.list_agents()

    # -- AI -------------------------------------------------------------------------------

    def ai_analyze_ticket(self, actor: AgentContext, ticket_id: str) -> Ticket:
        ticket = self.get_ticket(actor, ticket_id)
        desk_ai.require_ticket_ai_support(self._llm_client)
        customer = self._store.get_customer(ticket.customer_id)
        first_message = self._first_customer_message(ticket_id)
        result = desk_ai.analyze_ticket(
            self._llm_client,
            customer_name=customer.name if customer else "customer",
            subject=ticket.subject,
            message=first_message,
        )
        payload = result.payload
        self._store.update_ticket_fields(
            ticket_id,
            fields={
                "ai_summary": payload["summary"],
                "ai_key_points": payload["key_points"],
                "ai_category": payload["category"],
                "ai_suggested_priority": payload["priority"],
                "ai_urgency": payload["urgency"],
                "ai_sentiment": payload["sentiment"],
                "ai_analyzed_at": utc_now_iso(),
                "ai_provider": self._provider,
                "ai_model": self._model,
                "ai_tokens": result.usage.total_tokens if result.usage else None,
            },
        )
        metadata = {
            "category": payload["category"],
            "urgency": payload["urgency"],
            "sentiment": payload["sentiment"],
            "suggested_priority": payload["priority"],
        }
        self._event(ticket_id, actor, "ai_analyzed", metadata)
        self._audit(actor, "ai_analyzed", "ticket", ticket_id, metadata)
        return self.get_ticket(actor, ticket_id)

    def ai_suggest_reply(self, actor: AgentContext, ticket_id: str) -> Ticket:
        ticket = self.get_ticket(actor, ticket_id)
        desk_ai.require_ticket_ai_support(self._llm_client)
        customer = self._store.get_customer(ticket.customer_id)
        latest = self._latest_customer_message(ticket_id)
        result = desk_ai.suggest_reply(
            self._llm_client,
            customer_name=customer.name if customer else "customer",
            subject=ticket.subject,
            latest_message=latest,
        )
        self._store.update_ticket_fields(
            ticket_id,
            fields={
                "ai_reply": result.payload["reply"],
                "ai_reply_generated_at": utc_now_iso(),
                "ai_provider": self._provider,
                "ai_model": self._model,
            },
        )
        self._event(ticket_id, actor, "ai_reply_generated", {"draft": True})
        self._audit(actor, "ai_reply_generated", "ticket", ticket_id, {"draft": True})
        return self.get_ticket(actor, ticket_id)

    # -- history / audit ---------------------------------------------------------------------

    def ticket_history(self, actor: AgentContext, ticket_id: str) -> list[TicketEvent]:
        self.get_ticket(actor, ticket_id)
        return self._store.events_for_ticket(ticket_id)

    def audit_entries(self, actor: AgentContext, *, limit: int = 100, offset: int = 0) -> list[AuditEntry]:
        require_admin(actor)
        return self._store.list_audit(limit=min(max(limit, 1), 500), offset=max(offset, 0))

    # -- helpers ---------------------------------------------------------------------------------

    def _first_customer_message(self, ticket_id: str) -> str:
        messages = self._store.messages_for_ticket(ticket_id)
        for message in messages:
            if message.author_type == "customer":
                return message.body
        raise DeskValidation("ticket has no customer message to analyze")

    def _latest_customer_message(self, ticket_id: str) -> str:
        messages = [
            message for message in self._store.messages_for_ticket(ticket_id)
            if message.author_type == "customer"
        ]
        if not messages:
            raise DeskValidation("ticket has no customer message to reply to")
        return messages[-1].body

    @staticmethod
    def _content_hash(customer_id: str, message: str) -> str:
        normalized = f"{DUPLICATE_HASH_PREFIX}\n{customer_id}\n{normalize_text(message)}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_search(search: str | None) -> str | None:
        if search is None:
            return None
        cleaned = search.strip()
        return cleaned[:MAX_SEARCH_LENGTH] or None

    def _event(self, ticket_id: str, actor: AgentContext, event_type: str, metadata: dict) -> None:
        if event_type not in TICKET_EVENT_TYPES:
            raise DeskValidation(f"unknown event type {event_type!r}")
        self._store.insert_event(
            ticket_id=ticket_id, event_type=event_type, actor_id=actor.agent_id,
            metadata=metadata,
        )

    def _audit(self, actor: AgentContext, action: str, entity_type: str, entity_id: str,
               metadata: dict) -> None:
        if action not in AUDIT_ACTIONS:
            raise DeskValidation(f"unknown audit action {action!r}")
        self._store.insert_audit(
            action=action, entity_type=entity_type, entity_id=entity_id,
            actor_id=actor.agent_id, metadata=metadata,
        )
