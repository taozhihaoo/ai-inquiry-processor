"""Demo/seed data for the support desk — fictional, deterministic, offline.

Seeds agents, customers, tags and tickets with a spread of statuses,
priorities, tags and assignments, then runs the AI analysis + suggested-reply
flow (mock provider) on a few tickets. Everything is fictional; nothing calls
the real API.

Usage:
    python -m src.desk.seed                       # seed data/inquiries.db (skips if populated)
    python -m src.desk.seed --db path/to/db.sqlite
    python -m src.desk.seed --reset               # wipe desk tables, then seed
"""

from __future__ import annotations

import argparse
import sys

from src.desk.auth import AgentContext
from src.desk.service import DeskConflict, DeskService
from src.desk.storage import DeskStore
from src.llm_client import MockLLMClient

ADMIN_EMAIL = "admin@supportdesk.example"
AGENT_EMAILS = ("maya@supportdesk.example", "omar@supportdesk.example")

CUSTOMERS = (
    ("Alice Johnson", "alice.johnson@example.com", None),
    ("Bob Miller", "bob.miller@example.com", "acme-1042"),
    ("Carla Gomez", "carla.gomez@example.com", None),
    ("Dan Wu", "dan.wu@example.com", "globex-77"),
)

TAGS = ("billing", "login", "onboarding", "urgent", "feedback", "feature-request")

# (customer_email, subject, first_message, priority, tag_names, assignee_email, status)
TICKETS = (
    ("alice.johnson@example.com",
     "Cannot log in after password reset",
     "I cannot log into my account after resetting my password. The login page says my credentials are wrong even though I just reset them.",
     "high", ("login",), "maya@supportdesk.example", "open"),
    ("bob.miller@example.com",
     "Charged twice for September invoice",
     "I was charged twice for my September invoice. Please refund the duplicate payment of $49.",
     "high", ("billing",), None, "open"),
    ("carla.gomez@example.com",
     "Team locked out since this morning",
     "Our entire team has been locked out since this morning and we cannot process any orders. This is urgent, we are losing revenue every hour.",
     "urgent", ("login", "urgent"), "omar@supportdesk.example", "pending"),
    ("dan.wu@example.com",
     "Quote for enterprise plan",
     "Could you send me a quote for the Enterprise plan with 50 seats? We are evaluating vendors this month.",
     "normal", ("onboarding",), None, "open"),
    ("alice.johnson@example.com",
     "Mobile app crashes on upload",
     "The mobile app crashes every time I try to upload a photo. It worked fine last week.",
     "normal", (), "maya@supportdesk.example", "resolved"),
    ("bob.miller@example.com",
     "Feature idea: CSV export of tickets",
     "It would be great to export our tickets to CSV for our internal reporting. Keep up the great work, we love the product!",
     "low", ("feature-request", "feedback"), None, "closed"),
    ("carla.gomez@example.com",
     "How do I add a new team member?",
     "How do I add a new team member to our workspace? I could not find the option in settings.",
     "low", ("onboarding",), None, "open"),
    ("dan.wu@example.com",
     "Duplicate: charged twice for September",
     "I was charged twice for my September invoice. Please refund the duplicate payment of $49.",
     "high", (), None, "open"),
)

AI_ANALYZE_SUBJECTS = {
    "Cannot log in after password reset",
    "Team locked out since this morning",
    "Charged twice for September invoice",
}
AI_REPLY_SUBJECTS = {"Cannot log in after password reset"}


def seed_desk(db_path: str, *, reset: bool = False) -> dict:
    """Seed the desk; returns counts for CLI/CI verification. Fully offline."""
    store = DeskStore(db_path)
    store.initialize()
    if reset:
        _wipe_desk_tables(store)

    llm = MockLLMClient()
    service = DeskService(store=store, llm_client=llm, provider="mock", model="mock-1")

    existing_agents = store.list_agents()
    if existing_agents:
        return {"skipped": True, "agents": len(existing_agents),
                "customers": store.count_tickets(), "tickets": store.count_tickets()}

    admin, _ = store.insert_agent(
        name="Dana Reyes", email=ADMIN_EMAIL, role="admin", active=True
    )
    agents = {email: store.insert_agent(name=name, email=email, role="agent", active=True)[0]
              for name, email in (("Maya Patel", AGENT_EMAILS[0]),
                                  ("Omar Haddad", AGENT_EMAILS[1]))}
    customers = {email: store.insert_customer(name=name, email=email, external_id=external)[0]
                 for name, email, external in CUSTOMERS}
    tags = {name: store.insert_tag(name)[0] for name in TAGS}

    admin_context = AgentContext(agent_id=admin.id, name=admin.name, role=admin.role)

    created_tickets = []
    for (customer_email, subject, message, priority, _tag_names,
         _assignee_email, _status) in TICKETS:
        customer = customers[customer_email]
        try:
            ticket = service.create_ticket(
                admin_context, customer_id=customer.id, subject=subject,
                message=message, priority=priority,
            )
        except DeskConflict:
            # seeded duplicate (content-identical) -> link tags/status to the original
            duplicate = store.find_duplicate_ticket(
                service._content_hash(customer.id, message)  # noqa: SLF001 - seeding helper
            )
            if duplicate is not None:
                created_tickets.append(duplicate)
            continue
        created_tickets.append(ticket)

    for ticket, (_customer_email, _subject, _message, _priority, tag_names,
                 assignee_email, status) in zip(created_tickets, TICKETS, strict=False):
        for tag_name in tag_names:
            service.add_tag(admin_context, ticket.id, tag_id=tags[tag_name].id)
        if assignee_email:
            service.assign_ticket(
                admin_context, ticket.id, assignee_id=agents[assignee_email].id
            )
        if status != "open":
            service.update_ticket(admin_context, ticket.id, status=status)

    analyzed = 0
    for ticket in created_tickets:
        if ticket.subject in AI_ANALYZE_SUBJECTS:
            service.ai_analyze_ticket(admin_context, ticket.id)
            analyzed += 1
        if ticket.subject in AI_REPLY_SUBJECTS:
            service.ai_suggest_reply(admin_context, ticket.id)

    return {
        "skipped": False,
        "agents": 1 + len(agents),
        "customers": len(customers),
        "tags": len(tags),
        "tickets": len(created_tickets),
        "ai_analyzed": analyzed,
        "ai_replies": len(AI_REPLY_SUBJECTS),
    }


def _wipe_desk_tables(store: DeskStore) -> None:

    with store._connections.connect() as connection:  # noqa: SLF001 - own manager
        for table in ("audit_log", "ticket_events", "ticket_tags", "internal_notes",
                      "ticket_messages", "tickets", "tags", "agents", "customers"):
            connection.execute(f"DELETE FROM {table}")  # noqa: S608 - fixed table list


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.desk.seed",
        description="Seed fictional demo data for the AI customer support desk (offline).",
    )
    parser.add_argument(
        "--db", default="data/inquiries.db",
        help="SQLite database path (default: %(default)s)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="delete existing desk rows before seeding (inquiries are kept)",
    )
    args = parser.parse_args(argv)

    result = seed_desk(args.db, reset=args.reset)
    if result.get("skipped"):
        print(f"desk already populated ({result['agents']} agents) - nothing to do; "
              f"use --reset to reseed")
        return 0
    print(
        f"seeded: {result['agents']} agents, {result['customers']} customers, "
        f"{result['tags']} tags, {result['tickets']} tickets "
        f"({result['ai_analyzed']} AI-analyzed, {result['ai_replies']} AI drafts) -> {args.db}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
