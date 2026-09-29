"""REST API for the support desk (FastAPI router).

Thin transport layer: request validation (Pydantic), dev-auth resolution
(``X-Agent-Id`` header, see ``desk.auth``), delegation to :class:`DeskService`,
and mapping of service exceptions to safe JSON responses. No business logic,
no SQL, no provider calls here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.desk.auth import (
    DEV_AUTH_HEADER,
    AgentContext,
    AuthenticationError,
    AuthorizationError,
    require_admin,
    resolve_agent_context,
)
from src.desk.models import (
    TICKET_PRIORITIES,
    TICKET_STATUSES,
    Agent,
    Customer,
    Tag,
    Ticket,
)
from src.desk.service import DeskError, DeskService

# -- request schemas (API boundary only) -----------------------------------------


class CustomerIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=320)
    external_id: str | None = Field(default=None, max_length=200)

    @field_validator("name", "email")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("email")
    @classmethod
    def _email_shape(cls, value: str) -> str:
        if "@" not in value or value.count("@") != 1 or "." not in value.split("@")[1]:
            raise ValueError("must be a valid email address")
        return value


class TicketIn(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)
    subject: str = Field(min_length=1, max_length=300)
    message: str = Field(min_length=1, max_length=20_000)
    priority: str = Field(default="normal")

    @field_validator("subject", "message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class TicketPatch(BaseModel):
    subject: str | None = Field(default=None, min_length=1, max_length=300)
    status: str | None = Field(default=None)
    priority: str | None = Field(default=None)


class AssignIn(BaseModel):
    agent_id: str = Field(min_length=1, max_length=64)


class NoteIn(BaseModel):
    body: str = Field(min_length=1, max_length=10_000)

    @field_validator("body")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class MessageIn(BaseModel):
    body: str = Field(min_length=1, max_length=20_000)

    @field_validator("body")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class TagIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class TagLinkIn(BaseModel):
    tag_id: str = Field(min_length=1, max_length=64)


class AgentIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=320)
    role: str = Field(default="agent")

    @field_validator("email")
    @classmethod
    def _email_shape(cls, value: str) -> str:
        if "@" not in value or value.count("@") != 1 or "." not in value.split("@")[1]:
            raise ValueError("must be a valid email address")
        return value


# -- helpers -----------------------------------------------------------------------


def _error_response(status_code: int, code: str, message: str, **extra) -> JSONResponse:
    payload = {"error": {"code": code, "message": message}}
    payload["error"].update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def serialize_ticket(ticket: Ticket) -> dict:
    return ticket.to_dict()


def serialize_customer(customer: Customer) -> dict:
    return customer.to_dict()


def serialize_agent(agent: Agent) -> dict:
    return agent.to_dict()


def serialize_tag(tag: Tag) -> dict:
    return tag.to_dict()


def build_desk_router(service: DeskService) -> APIRouter:
    """Create the desk router bound to one service instance."""

    router = APIRouter(tags=["desk"])

    def current_agent(request: Request) -> AgentContext:
        """Development authentication: resolve X-Agent-Id to a stored agent."""
        header_value = request.headers.get(DEV_AUTH_HEADER)
        agent = None
        if header_value:
            agent = service.store.get_agent(header_value.strip())
        return resolve_agent_context(agent, header_value)

    def admin_agent(request: Request) -> AgentContext:
        context = current_agent(request)
        require_admin(context)
        return context

    # -- customers ---------------------------------------------------------------

    @router.post("/customers", status_code=201)
    def create_customer(body: CustomerIn, actor: AgentContext = Depends(current_agent)):
        customer, created = service.create_customer(
            actor, name=body.name, email=body.email, external_id=body.external_id
        )
        return JSONResponse(
            status_code=200 if not created else 201, content=serialize_customer(customer)
        )

    @router.get("/customers")
    def list_customers(
        actor: AgentContext = Depends(current_agent),
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        customers = service.list_customers(actor, search=search)
        return {
            "total": len(customers),
            "items": [serialize_customer(item) for item in customers],
        }

    @router.get("/customers/{customer_id}")
    def get_customer(customer_id: str, actor: AgentContext = Depends(current_agent)):
        return serialize_customer(service.get_customer(actor, customer_id))

    @router.get("/customers/{customer_id}/tickets")
    def customer_tickets(customer_id: str, actor: AgentContext = Depends(current_agent)):
        tickets = service.customer_tickets(actor, customer_id)
        return {
            "total": len(tickets),
            "items": [serialize_ticket(item) for item in tickets],
        }

    # -- tickets / inbox -----------------------------------------------------------

    @router.post("/tickets", status_code=201)
    def create_ticket(body: TicketIn, actor: AgentContext = Depends(current_agent)):
        if body.priority not in TICKET_PRIORITIES:
            return _error_response(
                422, "invalid_priority",
                f"priority must be one of {list(TICKET_PRIORITIES)}",
            )
        ticket = service.create_ticket(
            actor, customer_id=body.customer_id, subject=body.subject,
            message=body.message, priority=body.priority,
        )
        return JSONResponse(status_code=201, content=serialize_ticket(ticket))

    @router.get("/tickets")
    def list_tickets(  # noqa: PLR0913 - explicit query parameters
        actor: AgentContext = Depends(current_agent),
        status: str | None = None,
        priority: str | None = None,
        assignee: str | None = None,
        customer_id: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        statuses = [item.strip() for item in status.split(",")] if status else None
        priorities = [item.strip() for item in priority.split(",")] if priority else None
        for item in statuses or []:
            if item not in TICKET_STATUSES:
                return _error_response(
                    422, "invalid_status", f"status must be one of {list(TICKET_STATUSES)}"
                )
        unassigned = assignee == "none"
        assignee_id = None if unassigned or assignee is None else assignee
        tickets = service.list_tickets(
            actor,
            statuses=statuses, priorities=priorities, assignee_id=assignee_id,
            unassigned=unassigned, customer_id=customer_id, tag=tag,
            search=search, limit=limit, offset=offset,
        )
        return {"total": len(tickets), "items": [serialize_ticket(item) for item in tickets]}

    @router.get("/tickets/{ticket_id}")
    def ticket_detail(ticket_id: str, actor: AgentContext = Depends(current_agent)):
        return service.ticket_detail(actor, ticket_id).to_dict()

    @router.patch("/tickets/{ticket_id}")
    def update_ticket(
        ticket_id: str, body: TicketPatch, actor: AgentContext = Depends(current_agent)
    ):
        if body.status is not None and body.status not in TICKET_STATUSES:
            return _error_response(
                422, "invalid_status", f"status must be one of {list(TICKET_STATUSES)}"
            )
        if body.priority is not None and body.priority not in TICKET_PRIORITIES:
            return _error_response(
                422, "invalid_priority", f"priority must be one of {list(TICKET_PRIORITIES)}"
            )
        if body.status is None and body.priority is None and body.subject is None:
            return _error_response(422, "empty_update", "nothing to update")
        ticket = service.update_ticket(
            actor, ticket_id, status=body.status, priority=body.priority, subject=body.subject
        )
        return serialize_ticket(ticket)

    @router.post("/tickets/{ticket_id}/assign")
    def assign_ticket(
        ticket_id: str, body: AssignIn, actor: AgentContext = Depends(current_agent)
    ):
        return serialize_ticket(service.assign_ticket(actor, ticket_id, body.agent_id))

    @router.post("/tickets/{ticket_id}/unassign")
    def unassign_ticket(ticket_id: str, actor: AgentContext = Depends(current_agent)):
        return serialize_ticket(service.unassign_ticket(actor, ticket_id))

    @router.post("/tickets/{ticket_id}/messages", status_code=201)
    def add_message(
        ticket_id: str, body: MessageIn, actor: AgentContext = Depends(current_agent)
    ):
        message = service.add_customer_message(actor, ticket_id, body=body.body)
        return JSONResponse(status_code=201, content=message.to_dict())

    @router.post("/tickets/{ticket_id}/notes", status_code=201)
    def add_note(ticket_id: str, body: NoteIn, actor: AgentContext = Depends(current_agent)):
        note = service.add_note(actor, ticket_id, body=body.body)
        return JSONResponse(status_code=201, content=note.to_dict())

    @router.get("/tickets/{ticket_id}/notes")
    def list_notes(ticket_id: str, actor: AgentContext = Depends(current_agent)):
        notes = service.list_notes(actor, ticket_id)
        return {"total": len(notes), "items": [note.to_dict() for note in notes]}

    @router.get("/tickets/{ticket_id}/tags")
    def ticket_tags(ticket_id: str, actor: AgentContext = Depends(current_agent)):
        tags = service.ticket_tags(actor, ticket_id)
        return {"total": len(tags), "items": [serialize_tag(item) for item in tags]}

    @router.post("/tickets/{ticket_id}/tags", status_code=200)
    def add_ticket_tag(
        ticket_id: str, body: TagLinkIn, actor: AgentContext = Depends(current_agent)
    ):
        tag = service.add_tag(actor, ticket_id, tag_id=body.tag_id)
        return serialize_tag(tag)

    @router.delete("/tickets/{ticket_id}/tags/{tag_id}", status_code=204)
    def remove_ticket_tag(
        ticket_id: str, tag_id: str, actor: AgentContext = Depends(current_agent)
    ):
        service.remove_tag(actor, ticket_id, tag_id=tag_id)
        return JSONResponse(status_code=204, content=None)

    @router.get("/tickets/{ticket_id}/history")
    def ticket_history(ticket_id: str, actor: AgentContext = Depends(current_agent)):
        events = service.ticket_history(actor, ticket_id)
        return {"total": len(events), "items": [event.to_dict() for event in events]}

    # -- AI --------------------------------------------------------------------------

    @router.post("/tickets/{ticket_id}/ai/analyze")
    def ai_analyze(ticket_id: str, actor: AgentContext = Depends(current_agent)):
        ticket = service.ai_analyze_ticket(actor, ticket_id)
        return {
            "ticket_id": ticket.id,
            "note": "AI-generated classification/suggestions — may require human review.",
            "analysis": {
                "summary": ticket.ai_summary,
                "key_points": ticket.ai_key_points,
                "category": ticket.ai_category,
                "suggested_priority": ticket.ai_suggested_priority,
                "urgency": ticket.ai_urgency,
                "sentiment": ticket.ai_sentiment,
            },
            "provider": ticket.ai_provider,
            "model": ticket.ai_model,
            "analyzed_at": ticket.ai_analyzed_at,
        }

    @router.post("/tickets/{ticket_id}/ai/suggest-reply")
    def ai_suggest_reply(ticket_id: str, actor: AgentContext = Depends(current_agent)):
        ticket = service.ai_suggest_reply(actor, ticket_id)
        return {
            "ticket_id": ticket.id,
            "note": "AI-generated draft — review and edit before sending; never auto-sent.",
            "suggested_reply": ticket.ai_reply,
            "provider": ticket.ai_provider,
            "model": ticket.ai_model,
            "generated_at": ticket.ai_reply_generated_at,
        }

    # -- tags / agents ------------------------------------------------------------------

    @router.get("/tags")
    def list_tags(actor: AgentContext = Depends(current_agent)):
        tags = service.list_tags(actor)
        return {"total": len(tags), "items": [serialize_tag(item) for item in tags]}

    @router.post("/tags", status_code=201)
    def create_tag(body: TagIn, actor: AgentContext = Depends(admin_agent)):
        tag = service.create_tag(actor, name=body.name)
        return JSONResponse(status_code=201, content=serialize_tag(tag))

    @router.get("/agents")
    def list_agents(actor: AgentContext = Depends(current_agent)):
        agents = service.list_agents(actor)
        return {"total": len(agents), "items": [serialize_agent(item) for item in agents]}

    @router.post("/agents", status_code=201)
    def create_agent(body: AgentIn, actor: AgentContext = Depends(admin_agent)):
        agent, created = service.create_agent(
            actor, name=body.name, email=body.email, role=body.role
        )
        return JSONResponse(status_code=200 if not created else 201, content=serialize_agent(agent))

    @router.get("/agents/{agent_id}/tickets")
    def agent_tickets(agent_id: str, actor: AgentContext = Depends(current_agent)):
        tickets = service.agent_tickets(actor, agent_id)
        return {"total": len(tickets), "items": [serialize_ticket(item) for item in tickets]}

    # -- audit ----------------------------------------------------------------------------

    @router.get("/audit")
    def audit_log(
        actor: AgentContext = Depends(admin_agent),
        limit: int = 100,
        offset: int = 0,
    ):
        entries = service.audit_entries(actor, limit=limit, offset=offset)
        return {"total": len(entries), "items": [entry.to_dict() for entry in entries]}

    return router


def register_desk_exception_handlers(app) -> None:
    """Map desk business errors to safe JSON responses at app level."""

    @app.exception_handler(DeskError)
    async def _handle_desk(request: Request, error: DeskError) -> JSONResponse:
        code, status = error.code, error.status_code
        if error.metadata.get("code_override"):
            code = error.metadata["code_override"]
            status = error.metadata.get("status_code_override", error.status_code)
        extra = {
            key: value for key, value in error.metadata.items()
            if key not in ("code_override", "status_code_override")
        }
        return _error_response(status, code, str(error), **extra)

    @app.exception_handler(AuthenticationError)
    async def _handle_auth(request: Request, error: AuthenticationError) -> JSONResponse:
        return _error_response(401, "unauthorized", str(error))

    @app.exception_handler(AuthorizationError)
    async def _handle_forbidden(request: Request, error: AuthorizationError) -> JSONResponse:
        return _error_response(403, "forbidden", str(error))


__all__ = ["build_desk_router", "register_desk_exception_handlers"]
