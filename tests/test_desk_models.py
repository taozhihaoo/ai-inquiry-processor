"""Tests for desk domain models and the workflow transition rules."""

import pytest

from src.desk.models import (
    AGENT_ROLES,
    SENTIMENTS,
    TICKET_PRIORITIES,
    TICKET_STATUSES,
    URGENCY_LEVELS,
    DeskValidationError,
    new_id,
    normalize_text,
    validate_choice,
    validate_email,
)
from src.desk.workflow import InvalidTransitionError, validate_transition


class TestVocabularies:
    def test_ticket_enums_are_fixed(self):
        assert TICKET_STATUSES == ("open", "pending", "resolved", "closed")
        assert TICKET_PRIORITIES == ("low", "normal", "high", "urgent")
        assert URGENCY_LEVELS == ("low", "normal", "high", "urgent")
        assert SENTIMENTS == ("positive", "neutral", "negative")
        assert AGENT_ROLES == ("agent", "admin")

    def test_validate_choice_accepts_member(self):
        assert validate_choice("open", TICKET_STATUSES, "status") == "open"

    @pytest.mark.parametrize("value", ["OPEN", "done", "", "deleted"])
    def test_validate_choice_rejects_non_members(self, value):
        with pytest.raises(DeskValidationError, match="invalid status"):
            validate_choice(value, TICKET_STATUSES, "status")


class TestIdsAndNormalization:
    def test_new_id_is_32_hex_chars(self):
        value = new_id()
        assert len(value) == 32
        int(value, 16)  # must be hex

    def test_new_ids_are_unique(self):
        assert new_id() != new_id()

    def test_normalize_text_collapses_whitespace_and_case(self):
        assert normalize_text("  Hello   WORLD \n") == "hello world"

    def test_normalize_text_is_stable_for_duplicates(self):
        assert normalize_text("Charged TWICE!") == normalize_text("charged twice!")


class TestEmailValidation:
    def test_accepts_normal_address(self):
        assert validate_email("alice@example.com") == "alice@example.com"

    @pytest.mark.parametrize("bad", ["no-at-sign", "a@b", "two@@ats.com", "@example.com", ""])
    def test_rejects_malformed_address(self, bad):
        with pytest.raises(DeskValidationError, match="invalid email"):
            validate_email(bad)


class TestWorkflow:
    @pytest.mark.parametrize(
        ("current", "new"),
        [
            ("open", "pending"), ("open", "resolved"), ("open", "closed"),
            ("pending", "open"), ("pending", "resolved"),
            ("resolved", "open"), ("resolved", "closed"),
            ("closed", "open"),  # documented decision: closed is reopenable
        ],
    )
    def test_allowed_transitions(self, current, new):
        assert validate_transition(current, new) == new

    @pytest.mark.parametrize(
        ("current", "new"),
        [("open", "open"), ("pending", "pending"), ("resolved", "resolved"), ("closed", "closed")],
    )
    def test_same_status_is_noop(self, current, new):
        assert validate_transition(current, new) == new

    @pytest.mark.parametrize(
        ("current", "new"),
        [("pending", "closed"), ("closed", "pending"), ("closed", "resolved"), ("resolved", "pending")],
    )
    def test_forbidden_transitions_raise(self, current, new):
        with pytest.raises(InvalidTransitionError, match="cannot change ticket status"):
            validate_transition(current, new)

    def test_unknown_status_rejected(self):
        with pytest.raises(DeskValidationError, match="invalid status"):
            validate_transition("open", "done")
