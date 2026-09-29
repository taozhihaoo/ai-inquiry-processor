"""Migration tests: a Phase 2 database must keep working after the desk upgrade.

Old data is never touched; the desk schema only adds new tables (idempotent
``CREATE TABLE IF NOT EXISTS``). These tests pin that contract.
"""

import sqlite3

from src.desk.storage import DeskStore
from src.service import InquiryService
from src.storage import InquiryStore


def build_phase2_database(path):
    """Create a DB that only has the Phase 2 schema, with real data in it."""
    store = InquiryStore(path)
    store.initialize()
    service = InquiryService(
        llm_client=None, store=store, provider="mock", model="mock-1"
    )  # llm_client unused for pending inserts
    from src.storage import InquiryRecord

    record = InquiryRecord.pending(
        id="a" * 64, customer_name="Alice", message="Cannot log in.",
        provider="mock", model="mock-1",
    )
    store.insert_inquiry(record)
    store.create_run("run-phase2", provider="mock", model="mock-1")
    store.finish_run("run-phase2", total=1, succeeded=0, failed=1)
    return store, service


class TestBackwardCompatibility:
    def test_desk_initialize_adds_tables_and_keeps_phase2_data(self, tmp_path):
        db = tmp_path / "legacy.db"
        inquiry_store, _ = build_phase2_database(db)

        desk_store = DeskStore(db)
        desk_store.initialize()

        # old data still readable through the old store
        record = inquiry_store.get_inquiry("a" * 64)
        assert record is not None
        assert record.message == "Cannot log in."
        run = inquiry_store.get_run("run-phase2")
        assert run.failed == 1

        # new tables exist and are usable
        connection = sqlite3.connect(db)
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        connection.close()
        assert {
            "customers", "agents", "tags", "tickets", "ticket_messages",
            "internal_notes", "ticket_tags", "ticket_events", "audit_log",
        } <= tables

        customer, _ = desk_store.insert_customer(name="New", email="new@x.com")
        assert customer.id

    def test_repeated_initialize_is_idempotent(self, tmp_path):
        db = tmp_path / "legacy.db"
        build_phase2_database(db)
        desk_store = DeskStore(db)
        desk_store.initialize()
        desk_store.initialize()  # second run must not raise or duplicate
        assert desk_store.ping() is True

    def test_old_desk_data_survives_reinitialize(self, tmp_path):
        db = tmp_path / "desk.db"
        desk_store = DeskStore(db)
        desk_store.initialize()
        customer, _ = desk_store.insert_customer(name="Keep", email="keep@x.com")

        DeskStore(db).initialize()  # simulate upgrade re-run on existing desk DB

        assert desk_store.get_customer(customer.id).name == "Keep"

    def test_full_app_boots_on_phase2_database(self, tmp_path, mock_client):
        from fastapi.testclient import TestClient

        from src.api import create_app
        from src.desk.service import DeskService

        db = tmp_path / "legacy.db"
        inquiry_store, _ = build_phase2_database(db)
        desk_store = DeskStore(db)
        desk_store.initialize()
        inquiry_service = InquiryService(
            llm_client=mock_client, store=inquiry_store, provider="mock", model="mock-1"
        )
        desk_service = DeskService(
            store=desk_store, llm_client=mock_client, provider="mock", model="mock-1"
        )
        client = TestClient(create_app(service=inquiry_service, desk_service=desk_service))

        assert client.get("/health").json()["status"] == "ok"
        # legacy inquiry pipeline still answers on the same database
        response = client.post(
            "/inquiries",
            json={"customer_name": "Legacy User", "message": "hello"},
        )
        assert response.status_code == 201
