"""Tests for the SQLite persistence layer (all in-memory or tmp files, no network)."""

import sqlite3

import pytest

from src.storage import InquiryRecord, InquiryStore, StorageError

ID = "a" * 64
ID2 = "b" * 64


def make_store(tmp_path, name="t.db") -> InquiryStore:
    store = InquiryStore(tmp_path / name)
    store.initialize()
    return store


def pending_record(id=ID, **overrides):
    defaults = dict(
        id=id, customer_name="Alice", message="Cannot log in.",
        provider="mock", model="mock-1", run_id=None,
    )
    defaults.update(overrides)
    return InquiryRecord.pending(**defaults)


class TestSchemaAndInit:
    def test_initialize_is_idempotent(self, tmp_path):
        store = InquiryStore(tmp_path / "x.db")
        store.initialize()
        store.initialize()  # second call must not raise
        assert store.count_inquiries() == 0

    def test_creates_database_file_and_tables(self, tmp_path):
        make_store(tmp_path)
        assert (tmp_path / "t.db").is_file()
        connection = sqlite3.connect(tmp_path / "t.db")
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        connection.close()
        assert {"inquiries", "processing_runs"} <= tables

    def test_creates_parent_directories(self, tmp_path):
        store = InquiryStore(tmp_path / "deep" / "nested" / "x.db")
        store.initialize()
        assert (tmp_path / "deep" / "nested" / "x.db").is_file()


class TestInsertAndGet:
    def test_insert_and_get_roundtrip(self, tmp_path):
        store = make_store(tmp_path)
        assert store.insert_inquiry(pending_record()) is True
        record = store.get_inquiry(ID)
        assert record.status == "pending"
        assert record.customer_name == "Alice"
        assert record.created_at is not None
        assert record.updated_at is not None

    def test_duplicate_insert_returns_false(self, tmp_path):
        store = make_store(tmp_path)
        assert store.insert_inquiry(pending_record()) is True
        assert store.insert_inquiry(pending_record()) is False

    def test_missing_inquiry_returns_none(self, tmp_path):
        store = make_store(tmp_path)
        assert store.get_inquiry("missing") is None


class TestUpdates:
    def test_update_success_persists_analysis_and_usage(self, tmp_path):
        store = make_store(tmp_path)
        store.insert_inquiry(pending_record())
        store.update_success(
            ID, summary="Login problem.", category="Technical Support", priority="High",
            model="gpt-4o-mini", input_tokens=12, output_tokens=30, total_tokens=42,
            latency_ms=123.5, attempts=2, estimated_cost=0.000123,
        )
        record = store.get_inquiry(ID)
        assert record.status == "success"
        assert record.summary == "Login problem."
        assert record.category == "Technical Support"
        assert record.priority == "High"
        assert record.input_tokens == 12
        assert record.total_tokens == 42
        assert record.attempts == 2
        assert record.estimated_cost == 0.000123
        assert record.updated_at >= record.created_at

    def test_update_error_persists_message(self, tmp_path):
        store = make_store(tmp_path)
        store.insert_inquiry(pending_record())
        store.update_error(ID, error_message="LLMRetryableError: down", attempts=4)
        record = store.get_inquiry(ID)
        assert record.status == "error"
        assert record.error_message == "LLMRetryableError: down"
        assert record.attempts == 4
        assert record.summary is None

    def test_update_missing_inquiry_raises(self, tmp_path):
        store = make_store(tmp_path)
        with pytest.raises(StorageError, match="cannot update missing inquiry"):
            store.update_success(ID, summary="s", category="Sales", priority="Low")


class TestRuns:
    def test_create_and_finish_run(self, tmp_path):
        store = make_store(tmp_path)
        store.create_run("run-1", provider="mock", model="mock-1")
        run = store.get_run("run-1")
        assert run.provider == "mock"
        assert run.total is None  # not finished yet

        store.finish_run("run-1", total=3, succeeded=2, failed=1)
        run = store.get_run("run-1")
        assert (run.total, run.succeeded, run.failed) == (3, 2, 1)
        assert run.finished_at is not None

    def test_finish_missing_run_raises(self, tmp_path):
        store = make_store(tmp_path)
        with pytest.raises(StorageError, match="cannot finish missing run"):
            store.finish_run("ghost", total=0, succeeded=0, failed=0)

    def test_get_missing_run_returns_none(self, tmp_path):
        store = make_store(tmp_path)
        assert store.get_run("ghost") is None

    def test_inquiries_for_run(self, tmp_path):
        store = make_store(tmp_path)
        store.create_run("run-1", provider="mock")
        store.insert_inquiry(pending_record(id=ID, run_id="run-1"))
        store.insert_inquiry(pending_record(id=ID2, run_id="run-2"))
        assert [record.id for record in store.inquiries_for_run("run-1")] == [ID]


class TestFailureHandling:
    def test_unopenable_database_raises_storage_error(self, tmp_path):
        directory = tmp_path / "not-a-db"
        directory.mkdir()
        # make the parent a *file* so sqlite cannot create/open the database
        directory.rmdir()
        directory.write_text("this is not sqlite")
        bad_store = InquiryStore(directory)  # path exists as a file -> open fails
        with pytest.raises(StorageError):
            bad_store.initialize()

    def test_ping(self, tmp_path):
        assert make_store(tmp_path).ping() is True
        assert InquiryStore(tmp_path).ping() is False  # path is a tmp directory
