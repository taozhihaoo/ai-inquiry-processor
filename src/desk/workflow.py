"""Ticket workflow: the explicit, enforced set of status transitions.

Closed tickets CAN be reopened (closed -> open) — the simplest rule that
supports the real "customer came back" case; every transition is recorded in
history, so a reopen leaves a trail. Clients cannot set arbitrary status
strings: values must be in ``TICKET_STATUSES`` and pairs must be in the map
below, otherwise the change is rejected.
"""

from __future__ import annotations

from src.desk.models import TICKET_STATUSES, DeskValidationError

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "open": {"pending", "resolved", "closed"},
    "pending": {"open", "resolved"},
    "resolved": {"open", "closed"},
    "closed": {"open"},  # documented decision: reopening a closed ticket is allowed
}


class InvalidTransitionError(DeskValidationError):
    """Raised when a status change is not allowed by the workflow."""


def validate_transition(current: str, new: str) -> str:
    """Validate a status change; no-op (same status) is permitted to callers.

    Raises:
        DeskValidationError: if ``new`` is not a known status.
        InvalidTransitionError: if the pair is not an allowed transition.
    """
    if new not in TICKET_STATUSES:
        raise DeskValidationError(
            f"invalid status {new!r}; expected one of {list(TICKET_STATUSES)}"
        )
    if current == new:
        return current
    if new not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransitionError(
            f"cannot change ticket status from {current!r} to {new!r}; "
            f"allowed: {sorted(ALLOWED_TRANSITIONS[current])}"
        )
    return new
