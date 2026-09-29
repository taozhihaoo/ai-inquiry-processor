"""Tests for the FastAPI layer (in-process TestClient, mock provider, tmp SQLite)."""

import pytest

from src.llm_client import LLMRetryableError
from src.service import InquiryService
from src.storage import InquiryStore
from tests.test_service import FailingOnDemandClient

SAMPLE = {"customer_name": "Alice", "message": "I cannot log into my account."}


class TestHealth:
    def test_health_ok(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body == {"status": "ok", "database": "ok", "provider": "mock"}


class TestSubmitInquiry:
    def test_submit_returns_201_with_structured_result(self, api_client):
        response = api_client.post("/inquiries", json=SAMPLE)
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "success"
        assert body["category"] == "Technical Support"
        assert body["priority"] in ("Low", "Medium", "High")
        assert isinstance(body["summary"], str) and body["summary"]
        assert len(body["id"]) == 64
        assert body["duplicate"] is False

    def test_duplicate_submission_marked_and_not_reprocessed(self, api_client):
        first = api_client.post("/inquiries", json=SAMPLE)
        second = api_client.post("/inquiries", json=SAMPLE)
        assert second.status_code == 201
        assert second.json()["duplicate"] is True
        assert second.json()["id"] == first.json()["id"]
        assert second.json()["summary"] == first.json()["summary"]

    @pytest.mark.parametrize(
        "payload",
        [
            {},                                # missing fields
            {"customer_name": "", "message": "hi"},   # blank name
            {"customer_name": "Alice", "message": ""},  # blank message
            {"customer_name": "   ", "message": "hi"},  # whitespace-only name
            {"customer_name": "Alice"},        # missing message
            {"customer_name": "Alice", "message": None},
        ],
    )
    def test_validation_errors_return_422(self, api_client, payload):
        response = api_client.post("/inquiries", json=payload)
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_message_over_limit_returns_422(self, api_client):
        response = api_client.post(
            "/inquiries", json={"customer_name": "A", "message": "x" * 10_001}
        )
        assert response.status_code == 422


class TestGetInquiry:
    def test_get_by_id(self, api_client):
        created = api_client.post("/inquiries", json=SAMPLE).json()
        fetched = api_client.get(f"/inquiries/{created['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["id"] == created["id"]
        assert fetched.json()["created_at"] is not None

    def test_get_unknown_id_returns_404(self, api_client):
        response = api_client.get("/inquiries/" + "f" * 64)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


class TestBatch:
    def test_batch_success(self, api_client):
        payload = {"inquiries": [SAMPLE, {"customer_name": "Bob", "message": "Do you offer annual billing?"}]}
        response = api_client.post("/inquiries/batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert body["succeeded"] == 2
        assert body["failed"] == 0
        assert [item["row_number"] for item in body["results"]] == [1, 2]
        assert len(body["run_id"]) == 32

    def test_batch_partial_failure_does_not_crash(self, tmp_path):
        store = InquiryStore(tmp_path / "s.db")
        store.initialize()
        service = InquiryService(
            llm_client=FailingOnDemandClient(), store=store, provider="mock", model="mock-1"
        )
        from fastapi.testclient import TestClient

        from src.api import create_app

        client = TestClient(create_app(service=service))

        response = client.post(
            "/inquiries/batch",
            json={"inquiries": [
                {"customer_name": "A", "message": "good"},
                {"customer_name": "B", "message": "please fail"},
                {"customer_name": "C", "message": "good too"},
            ]},
        )
        assert response.status_code == 200  # partial failure is a normal outcome
        body = response.json()
        assert (body["succeeded"], body["failed"]) == (2, 1)
        failed_item = body["results"][1]
        assert failed_item["status"] == "error"
        assert "RateLimitError" in failed_item["error_message"]

    def test_batch_empty_list_returns_422(self, api_client):
        response = api_client.post("/inquiries/batch", json={"inquiries": []})
        assert response.status_code == 422

    def test_batch_deduplicates_repeated_content(self, api_client):
        response = api_client.post(
            "/inquiries/batch",
            json={"inquiries": [SAMPLE, dict(SAMPLE)]},
        )
        body = response.json()
        assert body["total"] == 2
        assert body["duplicates"] == 1
        assert body["succeeded"] == 2


class TestErrorSafety:
    def test_storage_failure_returns_opaque_500(self, tmp_path):
        """A broken database yields a safe JSON error — no traceback, no internals."""
        directory = tmp_path / "db-dir"
        directory.write_text("this is not sqlite")  # sqlite cannot open this path
        broken_store = InquiryStore(directory)
        service = InquiryService(
            llm_client=FailingOnDemandClient(), store=broken_store, provider="mock", model="m"
        )
        from fastapi.testclient import TestClient

        from src.api import create_app

        client = TestClient(create_app(service=service), raise_server_exceptions=False)
        response = client.post("/inquiries", json=SAMPLE)
        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "storage_error"
        assert body["error"]["message"] == "database operation failed"
        assert "Traceback" not in response.text
        assert "sqlite" not in response.text.lower()

    def test_provider_failure_recorded_as_error_row_not_crash(self, tmp_path):
        """Provider exceptions are isolated per row: the API still answers 2xx/201 shape."""

        class AlwaysFailsClient:
            def analyze_detailed(self, customer_name, message):
                raise LLMRetryableError("RateLimitError: quota exceeded", attempts=4)

        store = InquiryStore(tmp_path / "s.db")
        store.initialize()
        service = InquiryService(
            llm_client=AlwaysFailsClient(), store=store, provider="mock", model="m"
        )
        from fastapi.testclient import TestClient

        from src.api import create_app

        client = TestClient(create_app(service=service))
        response = client.post("/inquiries", json=SAMPLE)
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "error"
        assert "quota exceeded" in body["error_message"]
