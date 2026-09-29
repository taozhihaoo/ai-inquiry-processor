"""Application-layer integration tests: full pipeline, still fully offline.

1. CSV -> InquiryService -> Mock LLM -> SQLite -> JSON report on disk
2. HTTP request -> FastAPI -> InquiryService -> Mock LLM -> SQLite -> HTTP response
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.csv_loader import load_inquiries
from src.llm_client import MockLLMClient
from src.main import EXIT_OK, main
from src.service import InquiryService
from src.storage import InquiryStore
from tests.conftest import SAMPLE_CSV


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestCliToSqliteToJson:
    def test_csv_service_mock_sqlite_json_report(self, workdir):
        db_path = workdir / "data" / "cli.db"

        exit_code = main(
            ["--demo", "--input", str(SAMPLE_CSV), "--db", str(db_path),
             "--output-dir", str(workdir / "reports"), "--log-level", "ERROR"]
        )

        assert exit_code == EXIT_OK
        report = json.loads((workdir / "reports" / "inquiries_report.json").read_text(encoding="utf-8"))

        # report reflects the persisted run
        assert report["summary"]["total"] == len(load_inquiries(SAMPLE_CSV))
        assert report["summary"]["succeeded"] == report["summary"]["total"]
        assert report["report_metadata"]["run_id"]
        assert report["report_metadata"]["database"] == str(db_path)

        # every reported row is really persisted in SQLite
        store = InquiryStore(db_path)
        for row in report["results"]:
            record = store.get_inquiry(row["id"])
            assert record is not None
            assert record.status == "success"
            assert record.summary == row["summary"]

        run = store.get_run(report["report_metadata"]["run_id"])
        assert run.succeeded == run.total

        # CSV report exists with the same ids
        csv_lines = (workdir / "reports" / "inquiries_report.csv").read_text(encoding="utf-8").splitlines()
        assert len(csv_lines) == report["summary"]["total"] + 1  # header + rows

    def test_second_cli_run_is_all_duplicates(self, workdir):
        """Re-running the same CSV must not re-call the LLM (content identity)."""
        common = ["--demo", "--input", str(SAMPLE_CSV), "--db", str(workdir / "d.db"),
                  "--output-dir", str(workdir / "r"), "--log-level", "ERROR"]
        assert main(common) == EXIT_OK
        assert main(common) == EXIT_OK

        report = json.loads((workdir / "r" / "inquiries_report.json").read_text(encoding="utf-8"))
        assert report["summary"]["duplicates"] == report["summary"]["total"]


class TestHttpToSqlite:
    def test_http_request_reaches_sqlite_and_back(self, tmp_path):
        store = InquiryStore(tmp_path / "api.db")
        store.initialize()
        service = InquiryService(
            llm_client=MockLLMClient(), store=store, provider="mock", model="mock-1",
        )
        client = TestClient(create_app(service=service))

        # submit via HTTP
        created = client.post(
            "/inquiries",
            json={"customer_name": "Alice", "message": "I cannot log into my account."},
        )
        assert created.status_code == 201
        inquiry_id = created.json()["id"]

        # the data really is in SQLite
        record = store.get_inquiry(inquiry_id)
        assert record is not None
        assert record.status == "success"
        assert record.category == "Technical Support"

        # and readable back through the API
        fetched = client.get(f"/inquiries/{inquiry_id}")
        assert fetched.status_code == 200
        assert fetched.json()["summary"] == created.json()["summary"]

    def test_batch_run_is_queryable_from_storage(self, tmp_path):
        store = InquiryStore(tmp_path / "api.db")
        store.initialize()
        service = InquiryService(
            llm_client=MockLLMClient(), store=store, provider="mock", model="mock-1",
        )
        client = TestClient(create_app(service=service))

        batch = client.post(
            "/inquiries/batch",
            json={"inquiries": [
                {"customer_name": "A", "message": "one"},
                {"customer_name": "B", "message": "two"},
            ]},
        ).json()

        stored = store.inquiries_for_run(batch["run_id"])
        assert len(stored) == 2
        assert all(item.status == "success" for item in stored)
        run = store.get_run(batch["run_id"])
        assert run.finished_at is not None
