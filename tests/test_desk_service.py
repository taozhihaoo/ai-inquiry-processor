"""Tests for DeskService: workflow, RBAC, duplicate detection, history, audit, AI."""

import pytest

from src.desk.auth import AgentContext, AuthorizationError
from src.desk.service import (
    DeskConflict,
    DeskForbidden,
    DeskNotFound,
    DeskValidation,
)


@pytest.fixture
def ticket(desk_service, admin_ctx, desk_customer):
    return desk_service.create_ticket(
        admin_ctx, customer_id=desk_customer.id,
        subject="Cannot log in", message="I cannot log into my account.",
        priority="high",
    )


class TestCustomers:
    def test_create_and_get(self, desk_service, admin_ctx):
        customer, created = desk_service.create_customer(
            admin_ctx, name="Bob", email="BOB@Example.com"
        )
        assert created is True
        assert desk_service.get_customer(admin_ctx, customer.id).email == "bob@example.com"

    def test_get_unknown_raises(self, desk_service, admin_ctx):
        with pytest.raises(DeskNotFound):
            desk_service.get_customer(admin_ctx, "missing")

    def test_invalid_email_rejected(self, desk_service, admin_ctx):
        with pytest.raises(DeskValidation, match="invalid email"):
            desk_service.create_customer(admin_ctx, name="X", email="not-an-email")


class TestTicketCreation:
    def test_creates_open_ticket_with_first_message_and_history(
        self, desk_service, admin_ctx, desk_customer
    ):
        ticket = desk_service.create_ticket(
            admin_ctx, customer_id=desk_customer.id,
            subject="Help", message="Something broke.",
        )
        assert ticket.status == "open"
        assert ticket.assignee_id is None
        detail = desk_service.ticket_detail(admin_ctx, ticket.id)
        assert [message.body for message in detail.messages] == ["Something broke."]
        history = desk_service.ticket_history(admin_ctx, ticket.id)
        assert history[0].event_type == "ticket_created"

    def test_unknown_customer_rejected(self, desk_service, admin_ctx):
        with pytest.raises(DeskNotFound):
            desk_service.create_ticket(
                admin_ctx, customer_id="ghost", subject="S", message="M"
            )

    def test_invalid_priority_rejected(self, desk_service, admin_ctx, desk_customer):
        with pytest.raises(Exception, match="invalid ticket priority"):
            desk_service.create_ticket(
                admin_ctx, customer_id=desk_customer.id,
                subject="S", message="M", priority="asap",
            )

    def test_duplicate_content_conflicts_with_existing_ticket_id(
        self, desk_service, admin_ctx, desk_customer
    ):
        first = desk_service.create_ticket(
            admin_ctx, customer_id=desk_customer.id,
            subject="Charged twice", message="Please refund  the DUPLICATE payment.",
        )
        with pytest.raises(DeskConflict) as excinfo:
            desk_service.create_ticket(
                admin_ctx, customer_id=desk_customer.id,
                subject="Different subject", message="please refund the duplicate payment.",
            )
        assert excinfo.value.metadata["existing_ticket_id"] == first.id

    def test_same_text_from_different_customer_is_not_duplicate(
        self, desk_service, admin_ctx, desk_store, desk_customer
    ):
        other, _ = desk_store.insert_customer(name="Bea", email="bea@example.com")
        desk_service.create_ticket(
            admin_ctx, customer_id=desk_customer.id, subject="A", message="Same text."
        )
        second = desk_service.create_ticket(
            admin_ctx, customer_id=other.id, subject="B", message="Same text."
        )
        assert second.status == "open"

    def test_closed_ticket_content_can_be_resubmitted(
        self, desk_service, admin_ctx, desk_customer, ticket
    ):
        desk_service.update_ticket(admin_ctx, ticket.id, status="closed")
        retry = desk_service.create_ticket(
            admin_ctx, customer_id=desk_customer.id,
            subject="Again", message="I cannot log into my account.",
        )
        assert retry.status == "open"


class TestWorkflow:
    def test_valid_transition_updates_status_and_history(self, desk_service, admin_ctx, ticket):
        updated = desk_service.update_ticket(admin_ctx, ticket.id, status="pending")
        assert updated.status == "pending"
        events = [event.event_type for event in desk_service.ticket_history(admin_ctx, ticket.id)]
        assert "status_changed" in events

    def test_resolved_sets_timestamp_and_reopen_clears_it(self, desk_service, admin_ctx, ticket):
        resolved = desk_service.update_ticket(admin_ctx, ticket.id, status="resolved")
        assert resolved.resolved_at is not None
        reopened = desk_service.update_ticket(admin_ctx, ticket.id, status="open")
        assert reopened.resolved_at is None

    def test_invalid_transition_conflicts(self, desk_service, admin_ctx, ticket):
        desk_service.update_ticket(admin_ctx, ticket.id, status="closed")
        with pytest.raises(DeskConflict, match="cannot change ticket status"):
            desk_service.update_ticket(admin_ctx, ticket.id, status="pending")

    def test_priority_update_recorded(self, desk_service, admin_ctx, ticket):
        updated = desk_service.update_ticket(admin_ctx, ticket.id, priority="urgent")
        assert updated.priority == "urgent"
        events = desk_service.ticket_history(admin_ctx, ticket.id)
        assert events[-1].event_type == "priority_changed"
        assert events[-1].metadata == {"from": "high", "to": "urgent"}


class TestAssignment:
    def test_assign_and_unassign(self, desk_service, admin_ctx, agent_ctx, ticket):
        assigned = desk_service.assign_ticket(admin_ctx, ticket.id, agent_ctx.agent_id)
        assert assigned.assignee_id == agent_ctx.agent_id
        unassigned = desk_service.unassign_ticket(admin_ctx, ticket.id)
        assert unassigned.assignee_id is None
        events = [event.event_type for event in desk_service.ticket_history(admin_ctx, ticket.id)]
        assert "assigned" in events and "unassigned" in events

    def test_assign_to_unknown_agent_404(self, desk_service, admin_ctx, ticket):
        with pytest.raises(DeskNotFound):
            desk_service.assign_ticket(admin_ctx, ticket.id, "ghost")

    def test_agent_cannot_modify_ticket_assigned_to_other(
        self, desk_service, desk_store, admin_ctx, ticket
    ):
        rival, _ = desk_store.insert_agent(name="Rival", email="rival@x.com", role="agent")
        desk_service.assign_ticket(admin_ctx, ticket.id, rival.id)
        outsider, _ = desk_store.insert_agent(name="Out", email="out@x.com", role="agent")
        outsider_ctx = AgentContext(agent_id=outsider.id, name=outsider.name, role="agent")
        with pytest.raises(DeskForbidden):
            desk_service.update_ticket(outsider_ctx, ticket.id, status="pending")

    def test_agent_can_modify_unassigned_ticket(self, desk_service, agent_ctx, ticket):
        updated = desk_service.update_ticket(agent_ctx, ticket.id, priority="low")
        assert updated.priority == "low"

    def test_agent_cannot_reassign_ticket_owned_by_other(
        self, desk_service, desk_store, admin_ctx, ticket
    ):
        rival, _ = desk_store.insert_agent(name="Rival", email="rival@x.com", role="agent")
        outsider, _ = desk_store.insert_agent(name="Out", email="out@x.com", role="agent")
        desk_service.assign_ticket(admin_ctx, ticket.id, rival.id)
        outsider_ctx = AgentContext(agent_id=outsider.id, name=outsider.name, role="agent")
        with pytest.raises(DeskForbidden):
            desk_service.assign_ticket(outsider_ctx, ticket.id, outsider_ctx.agent_id)

    def test_admin_can_modify_any_ticket(self, desk_service, desk_store, admin_ctx, ticket):
        other, _ = desk_store.insert_agent(name="Other", email="other@x.com", role="agent")
        desk_service.assign_ticket(admin_ctx, ticket.id, other.id)
        updated = desk_service.update_ticket(admin_ctx, ticket.id, status="resolved")
        assert updated.status == "resolved"


class TestNotesAndMessages:
    def test_note_has_author_and_event(self, desk_service, agent_ctx, ticket):
        note = desk_service.add_note(agent_ctx, ticket.id, body="Checked logs.")
        assert note.author_name == "Sam Agent"
        events = [
            event.event_type for event in desk_service.ticket_history(agent_ctx, ticket.id)
        ]
        assert "note_added" in events
        assert [item.body for item in desk_service.list_notes(agent_ctx, ticket.id)] == ["Checked logs."]

    def test_customer_message_bumps_ticket(self, desk_service, agent_ctx, ticket):
        message = desk_service.add_customer_message(agent_ctx, ticket.id, body="Still broken.")
        assert message.author_type == "customer"
        messages = desk_service.ticket_detail(agent_ctx, ticket.id).messages
        assert [item.body for item in messages] == [
            "I cannot log into my account.", "Still broken.",
        ]


class TestTags:
    def test_admin_creates_tag_and_agents_link_it(
        self, desk_service, admin_ctx, agent_ctx, ticket
    ):
        tag = desk_service.create_tag(admin_ctx, name="billing")
        linked = desk_service.add_tag(agent_ctx, ticket.id, tag_id=tag.id)
        assert linked.name == "billing"
        assert [item.name for item in desk_service.ticket_tags(agent_ctx, ticket.id)] == ["billing"]

    def test_agent_cannot_create_tag(self, desk_service, agent_ctx):
        with pytest.raises(AuthorizationError):
            desk_service.create_tag(agent_ctx, name="nope")

    def test_adding_unknown_tag_404(self, desk_service, agent_ctx, ticket):
        with pytest.raises(DeskNotFound):
            desk_service.add_tag(agent_ctx, ticket.id, tag_id="ghost")

    def test_duplicate_link_is_noop(self, desk_service, admin_ctx, ticket):
        tag = desk_service.create_tag(admin_ctx, name="vip")
        desk_service.add_tag(admin_ctx, ticket.id, tag_id=tag.id)
        desk_service.add_tag(admin_ctx, ticket.id, tag_id=tag.id)  # no error, single link
        assert len(desk_service.ticket_tags(admin_ctx, ticket.id)) == 1

    def test_remove_tag(self, desk_service, admin_ctx, ticket):
        tag = desk_service.create_tag(admin_ctx, name="temp")
        desk_service.add_tag(admin_ctx, ticket.id, tag_id=tag.id)
        desk_service.remove_tag(admin_ctx, ticket.id, tag_id=tag.id)
        assert desk_service.ticket_tags(admin_ctx, ticket.id) == []


class TestAgentsAndAudit:
    def test_only_admin_creates_agents(self, desk_service, admin_ctx, agent_ctx):
        agent, created = desk_service.create_agent(
            admin_ctx, name="New", email="new@x.com", role="agent"
        )
        assert created
        with pytest.raises(AuthorizationError):
            desk_service.create_agent(agent_ctx, name="Nope", email="nope@x.com", role="admin")

    def test_audit_log_admin_only_and_records_actions(
        self, desk_service, admin_ctx, agent_ctx, ticket
    ):
        desk_service.update_ticket(agent_ctx, ticket.id, status="pending")
        entries = desk_service.audit_entries(admin_ctx)
        actions = {entry.action for entry in entries}
        assert {"ticket_created", "status_changed"} <= actions
        with pytest.raises(AuthorizationError):
            desk_service.audit_entries(agent_ctx)

    def test_audit_metadata_is_scoped(self, desk_service, admin_ctx, desk_customer):
        desk_service.create_customer(admin_ctx, name="P", email="p@x.com")
        entry = desk_service.audit_entries(admin_ctx)[0]
        assert entry.metadata.keys() <= {"email"}  # only whitelisted fields
        assert "password" not in str(entry.metadata).lower()


class TestAI:
    def test_analyze_stores_advisory_fields(
        self, desk_service, admin_ctx, desk_customer, ticket
    ):
        analyzed = desk_service.ai_analyze_ticket(admin_ctx, ticket.id)
        assert analyzed.ai_category in ("Sales", "Technical Support", "Billing", "General Question")
        assert analyzed.ai_urgency in ("low", "normal", "high", "urgent")
        assert analyzed.ai_sentiment in ("positive", "neutral", "negative")
        assert analyzed.ai_summary
        assert analyzed.ai_key_points
        assert analyzed.ai_provider == "mock"
        events = [event.event_type for event in desk_service.ticket_history(admin_ctx, ticket.id)]
        assert "ai_analyzed" in events

    def test_analyze_does_not_change_workflow_fields(
        self, desk_service, admin_ctx, desk_customer, ticket
    ):
        analyzed = desk_service.ai_analyze_ticket(admin_ctx, ticket.id)
        assert analyzed.status == ticket.status
        assert analyzed.priority == ticket.priority
        assert analyzed.assignee_id == ticket.assignee_id

    def test_suggest_reply_uses_latest_customer_message(
        self, desk_service, admin_ctx, agent_ctx, ticket
    ):
        desk_service.add_customer_message(agent_ctx, ticket.id, body="Still can't log in!")
        drafted = desk_service.ai_suggest_reply(admin_ctx, ticket.id)
        assert drafted.ai_reply
        assert "Still can't log in!" in drafted.ai_reply  # deterministic mock references input
        assert drafted.ai_reply_generated_at is not None

    def test_suggest_reply_is_draft_only_never_a_message(
        self, desk_service, admin_ctx, ticket
    ):
        desk_service.ai_suggest_reply(admin_ctx, ticket.id)
        messages = desk_service.ticket_detail(admin_ctx, ticket.id).messages
        assert len(messages) == 1  # only the original customer message
        assert messages[0].author_type == "customer"


class TestInboxQueries:
    @pytest.fixture
    def inbox(self, desk_service, desk_store, admin_ctx, agent_ctx, desk_customer):
        other, _ = desk_store.insert_customer(name="Bob", email="bob@x.com")
        t1 = desk_service.create_ticket(
            admin_ctx, customer_id=desk_customer.id, subject="One", message="m"
        )
        t2 = desk_service.create_ticket(
            admin_ctx, customer_id=other.id, subject="Two", message="m", priority="urgent"
        )
        desk_service.assign_ticket(admin_ctx, t1.id, agent_ctx.agent_id)
        desk_service.update_ticket(admin_ctx, t2.id, status="resolved")
        return {"t1": t1, "t2": t2}

    def test_filter_by_status(self, desk_service, admin_ctx, inbox):
        open_ids = {
            item.id for item in desk_service.list_tickets(admin_ctx, statuses=["open"])
        }
        assert open_ids == {inbox["t1"].id}

    def test_filter_unassigned(self, desk_service, admin_ctx, inbox):
        assert {
            item.id for item in desk_service.list_tickets(admin_ctx, unassigned=True)
        } == {inbox["t2"].id}

    def test_filter_assignee(self, desk_service, admin_ctx, agent_ctx, inbox):
        assert {
            item.id for item in desk_service.list_tickets(
                admin_ctx, assignee_id=agent_ctx.agent_id
            )
        } == {inbox["t1"].id}

    def test_high_priority_view(self, desk_service, admin_ctx, inbox):
        urgent = desk_service.list_tickets(admin_ctx, priorities=["urgent"])
        assert [item.id for item in urgent] == [inbox["t2"].id]
