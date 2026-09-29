"""Tests for the demo seed: runs offline, is verifiable, and is idempotent."""

import sqlite3

from src.desk.seed import seed_desk
from src.desk.storage import DeskStore


class TestSeed:
    def test_seed_creates_full_demo_dataset(self, tmp_path):
        db = tmp_path / "seed.db"
        result = seed_desk(str(db))

        assert result["skipped"] is False
        assert result["agents"] >= 3  # 1 admin + 2 agents
        assert result["customers"] == 4
        assert result["tags"] == 6
        assert result["tickets"] == 8
        assert result["ai_analyzed"] == 3
        assert result["ai_replies"] == 1

        store = DeskStore(db)
        tickets = store.list_tickets(limit=200)
        assert len(tickets) == 8
        statuses = {ticket.status for ticket in tickets}
        assert {"open", "pending", "resolved", "closed"} <= statuses
        priorities = {ticket.priority for ticket in tickets}
        assert {"low", "normal", "high", "urgent"} <= priorities

    def test_seeded_ai_fields_are_populated(self, tmp_path):
        db = tmp_path / "seed.db"
        seed_desk(str(db))
        store = DeskStore(db)
        analyzed = [t for t in store.list_tickets(limit=200) if t.ai_summary]
        assert len(analyzed) == 3
        drafted = [t for t in store.list_tickets(limit=200) if t.ai_reply]
        assert len(drafted) == 1
        assert all(t.ai_provider == "mock" for t in analyzed)

    def test_seed_is_idempotent_second_run_skips(self, tmp_path):
        db = tmp_path / "seed.db"
        seed_desk(str(db))
        second = seed_desk(str(db))
        assert second["skipped"] is True
        store = DeskStore(db)
        assert len(store.list_tickets(limit=200)) == 8  # no duplicates created

    def test_reset_reseeds_from_scratch(self, tmp_path):
        db = tmp_path / "seed.db"
        seed_desk(str(db))
        result = seed_desk(str(db), reset=True)
        assert result["skipped"] is False
        store = DeskStore(db)
        assert len(store.list_tickets(limit=200)) == 8
        # audit log was wiped by reset and rebuilt by the fresh seeding
        assert store.list_audit()

    def test_seed_coexists_with_inquiry_tables(self, tmp_path):
        db = tmp_path / "shared.db"
        from src.storage import InquiryStore

        inquiry_store = InquiryStore(db)
        inquiry_store.initialize()
        seed_desk(str(db))

        connection = sqlite3.connect(db)
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        connection.close()
        assert {"inquiries", "processing_runs"} <= tables
        assert {"tickets", "customers"} <= tables

    def test_no_real_personal_data(self, tmp_path):
        """Every seeded email is on the fictional example.com reserve."""
        db = tmp_path / "seed.db"
        seed_desk(str(db))
        store = DeskStore(db)
        for customer in store.list_customers():
            assert customer.email.endswith("@example.com")
