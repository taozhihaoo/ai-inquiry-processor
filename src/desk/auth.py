"""Development authentication + minimal RBAC for the support desk.

There is deliberately NO production authentication (no OAuth/JWT/SSO): this is
a portfolio-scale internal tool. The mechanism is an explicit, honest dev
scheme — the caller identifies itself with the ``X-Agent-Id`` header naming a
seeded agent, and the server resolves that agent from the database:

- unknown id, missing header or inactive agent  -> 401
- authenticated but lacking the required role   -> 403

The authorization model is enforced in the service layer (``DeskService``
receives an :class:`AgentContext` with every call), not in the API layer alone,
so CLI/seed callers use the exact same rules. README states clearly that this
is demo/development authentication, not production identity management.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.desk.models import AGENT_ROLES, Agent, DeskValidationError, validate_choice

DEV_AUTH_HEADER = "X-Agent-Id"


class AuthenticationError(Exception):
    """401 — request did not identify a known, active agent."""


class AuthorizationError(Exception):
    """403 — authenticated agent lacks the required role."""


@dataclass(frozen=True)
class AgentContext:
    """The authenticated actor for one desk operation."""

    agent_id: str
    name: str
    role: str

    @classmethod
    def from_agent(cls, agent: Agent) -> AgentContext:
        return cls(agent_id=agent.id, name=agent.name, role=agent.role)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def resolve_agent_context(agent: Agent | None, header_value: str | None) -> AgentContext:
    """Turn the dev-auth header + stored agent into a context (or raise 401 causes)."""
    if not header_value:
        raise AuthenticationError(
            f"missing {DEV_AUTH_HEADER} header (development authentication; see README)"
        )
    if agent is None:
        raise AuthenticationError(f"unknown agent id in {DEV_AUTH_HEADER} header")
    if not agent.active:
        raise AuthenticationError("this agent account is deactivated")
    return AgentContext.from_agent(agent)


def validate_agent_role(role: str) -> str:
    try:
        return validate_choice(role, AGENT_ROLES, "agent role")
    except DeskValidationError as exc:
        raise DeskValidationError(str(exc)) from None


def require_admin(context: AgentContext) -> None:
    if not context.is_admin:
        raise AuthorizationError("this action requires the 'admin' role")
