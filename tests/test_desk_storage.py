"""Tests for the desk persistence layer (customers/agents/tags/tickets/history/audit)."""

import pytest

from src.desk.models import Ticket, new_id
from src.storage import StorageError, utc_now_iso


def make_ticket(customer_id: str, **overrides) -> Ticket:
    now = utc_now_iso()
    defaults = dict(
        id=new_id(), customer_id=customer_id, subject="Subject", status="open",
        priority="normal", assignee_id=None, content_hash="hash-" + new_id()[:8],
        created_at=now, updated_at=now,
    )
    defaults.update(overrides)
    return Ticket(**defaults)


@pytest.fixture
def store(desk_store, admin_ctx, agent_ctx, desk_customer):
    return desk_store


@pytest.fixture
def customer_id(desk_customer):
    return desk_customer.id


class TestCustomers:
    def test_insert_and_get(self, store):
        customer, created = store.insert_customer(name="Bob", email="bob@x.com")
        assert created is True
        fetched = store.get_customer(customer.id)
        assert fetched.email == "bob@x.com"
        assert fetched.created_at is not None

    def test_duplicate_email_returns_existing(self, store):
        first, created1 = store.insert_customer(name="Bob", email="bob@x.com")
        second, created2 = store.insert_customer(name="Bobby", email="bob@x.com")
        assert created1 is True and created2 is False
        assert second.id == first.id

    def test_list_with_search_matches_name_and_email(self, store):
        store.insert_customer(name="Alice Wonder", email="alice@wonder.com")
        store.insert_customer(name="Bob", email="bob@mail.com")
        by_name = {c.email for c in store.list_customers(search="wonder")}
        by_email = {c.email for c in store.list_customers(search="mail.com")}
        assert by_name == {"alice@wonder.com"}
        assert by_email == {"bob@mail.com"}

    def test_search_escapes_like_wildcards(self, store):
        store.insert_customer(name="100% Pure", email="pure@x.com")
        store.insert_customer(name="Purely Normal", email="normal@x.com")
        matches = {c.name for c in store.list_customers(search="100%")}
        assert matches == {"100% Pure"}


class TestAgents:
    def test_insert_and_role_check(self, store):
        agent, created = store.insert_agent(name="A", email="a@x.com", role="admin")
        assert created and store.get_agent(agent.id).role == "admin"

    def test_duplicate_email_returns_existing(self, store):
        store.insert_agent(name="A", email="a@x.com", role="agent")
        _, created = store.insert_agent(name="A2", email="a@x.com", role="admin")
        assert created is False

    def test_inactive_agents_excluded_from_active_listing(self, store):
        active_before = len(store.list_agents(active_only=True))
        store.insert_agent(name="A", email="a@x.com", role="agent", active=True)
        store.insert_agent(name="B", email="b@x.com", role="agent", active=False)
        active = store.list_agents(active_only=True)
        assert len(active) == active_before + 1
        assert all(agent.active for agent in active)


class TestTags:
    def test_unique_names_case_insensitive(self, store):
        tag, created = store.insert_tag("Billing")
        assert created
        same, created2 = store.insert_tag("billing")
        assert created2 is False
        assert same.id == tag.id


class TestTickets:
    def test_insert_get_and_duplicate_lookup(self, store, customer_id):
        ticket = make_ticket(customer_id, content_hash="h-1")
        assert store.insert_ticket(ticket) is True
        assert store.get_ticket(ticket.id).subject == "Subject"
        assert store.find_duplicate_ticket("h-1").id == ticket.id
        assert store.find_duplicate_ticket("missing") is None

    def test_duplicate_lookup_ignores_closed_tickets(self, store, customer_id):
        ticket = make_ticket(customer_id, content_hash="h-closed", status="closed")
        store.insert_ticket(ticket)
        assert store.find_duplicate_ticket("h-closed") is None

    def test_update_fields_allowlist_and_updated_at_bump(self, store, customer_id):
        ticket = make_ticket(customer_id)
        store.insert_ticket(ticket)
        before = store.get_ticket(ticket.id).updated_at
        store.update_ticket_fields(ticket.id, fields={"priority": "high"})
        updated = store.get_ticket(ticket.id)
        assert updated.priority == "high"
        assert updated.updated_at >= before

    def test_update_fields_rejects_unknown_columns(self, store, customer_id):
        ticket = make_ticket(customer_id)
        store.insert_ticket(ticket)
        with pytest.raises(StorageError, match="cannot update ticket field"):
            store.update_ticket_fields(ticket.id, fields={"status_sql": "x"})

    def test_ai_key_points_roundtrip_as_json(self, store, customer_id):
        ticket = make_ticket(customer_id)
        store.insert_ticket(ticket)
        store.update_ticket_fields(
            ticket.id, fields={"ai_key_points": ["login broken", "since Monday"]}
        )
        assert store.get_ticket(ticket.id).ai_key_points == ["login broken", "since Monday"]

    def test_resolved_at_roundtrip(self, store, customer_id):
        ticket = make_ticket(customer_id)
        store.insert_ticket(ticket)
        store.update_ticket_fields(ticket.id, fields={"resolved_at": utc_now_iso()})
        assert store.get_ticket(ticket.id).resolved_at is not None
        store.update_ticket_fields(ticket.id, fields={"resolved_at": None})
        assert store.get_ticket(ticket.id).resolved_at is None


class TestTicketListing:
    @pytest.fixture
    def populated(self, store, customer_id):
        second_customer, _ = store.insert_customer(name="Bob", email="bob@x.com")
        open_ticket = make_ticket(customer_id, status="open", priority="high")
        pending = make_ticket(customer_id, status="pending", priority="low", subject="Pending one")
        other = make_ticket(second_customer.id, subject="Bob issue", status="closed")
        for ticket in (open_ticket, pending, other):
            store.insert_ticket(ticket)
        tag, _ = store.insert_tag("billing")
        store.add_ticket_tag(open_ticket.id, tag.id)
        return {"open": open_ticket, "pending": pending, "other": other, "tag": tag}

    def ids(self, tickets):
        return {ticket.id for ticket in tickets}

    def test_filter_status(self, store, populated):
        assert self.ids(store.list_tickets(statuses=["open"])) == {populated["open"].id}

    def test_filter_priority(self, store, populated):
        assert self.ids(store.list_tickets(priorities=["high"])) == {populated["open"].id}

    def test_filter_unassigned(self, store, populated):
        assert store.count_tickets(unassigned=True) == 3

    def test_filter_customer(self, store, populated):
        assert store.count_tickets(customer_id=populated["other"].customer_id) == 1

    def test_filter_tag(self, store, populated):
        assert self.ids(store.list_tickets(tag="billing")) == {populated["open"].id}

    def test_search_across_subject(self, store, populated):
        assert self.ids(store.list_tickets(search="pending")) == {populated["pending"].id}

    def test_search_across_customer_name(self, store, populated):
        assert store.count_tickets(search="Bob") == 1

    def test_search_wildcards_escaped(self, store, customer_id):
        store.insert_ticket(make_ticket(customer_id, subject="Save 50% now"))
        assert store.count_tickets(search="50%") == 1
        assert store.count_tickets(search="100%") == 0

    def test_pagination(self, store, populated):
        assert len(store.list_tickets(limit=2)) == 2
        assert len(store.list_tickets(limit=2, offset=2)) == 1


class TestMessagesNotesTagsEventsAudit:
    def test_message_flow(self, store, customer_id):
        ticket = make_ticket(customer_id)
        store.insert_ticket(ticket)
        from src.desk.models import TicketMessage

        message = TicketMessage(
            id=new_id(), ticket_id=ticket.id, author_type="customer",
            author_name="Alice", body="It broke.", created_at=utc_now_iso(),
        )
        assert store.insert_message(message) is True
        assert store.messages_for_ticket(ticket.id)[0].body == "It broke."

    def test_notes_flow(self, store, customer_id):
        ticket = make_ticket(customer_id)
        store.insert_ticket(ticket)
        from src.desk.models import InternalNote

        note = InternalNote(
            id=new_id(), ticket_id=ticket.id, author_id="agent-1",
            author_name="Sam", body="Retrying creds.", created_at=utc_now_iso(),
        )
        store.insert_note(note)
        assert store.notes_for_ticket(ticket.id)[0].body == "Retrying creds."

    def test_tag_link_and_unlink(self, store, customer_id):
        ticket = make_ticket(customer_id)
        store.insert_ticket(ticket)
        tag, _ = store.insert_tag("vip")
        assert store.add_ticket_tag(ticket.id, tag.id) is True
        assert store.add_ticket_tag(ticket.id, tag.id) is False  # idempotent
        assert [item.name for item in store.tags_for_ticket(ticket.id)] == ["vip"]
        assert store.remove_ticket_tag(ticket.id, tag.id) is True
        assert store.remove_ticket_tag(ticket.id, tag.id) is False
        assert store.tags_for_ticket(ticket.id) == []

    def test_events_flow(self, store, customer_id):
        ticket = make_ticket(customer_id)
        store.insert_ticket(ticket)
        store.insert_event(ticket_id=ticket.id, event_type="ticket_created",
                           actor_id=None, metadata={"subject": "S"})
        store.insert_event(ticket_id=ticket.id, event_type="status_changed",
                           actor_id="a1", metadata={"from": "open", "to": "pending"})
        events = store.events_for_ticket(ticket.id)
        assert [event.event_type for event in events] == ["ticket_created", "status_changed"]
        assert events[1].metadata == {"from": "open", "to": "pending"}

    def test_audit_flow(self, store):
        store.insert_audit(action="customer_created", entity_type="customer",
                           entity_id="c1", actor_id=None, metadata={"email": "x@y.z"})
        entries = store.list_audit()
        assert entries[0].action == "customer_created"
        assert entries[0].metadata == {"email": "x@y.z"}


class TestForeignKeys:
    def test_ticket_requires_existing_customer(self, store):
        from src.desk.models import DeskValidationError  # noqa: F401

        with pytest.raises(StorageError, match="database operation failed"):
            store.insert_ticket(make_ticket("ghost-customer"))

    def test_tag_link_requires_existing_tag(self, store, customer_id):
        ticket = make_ticket(customer_id)
        store.insert_ticket(ticket)
        with pytest.raises(StorageError, match="database operation failed"):
            store.add_ticket_tag(ticket.id, "ghost-tag")
