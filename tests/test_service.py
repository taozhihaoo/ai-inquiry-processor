"""Tests for the InquiryService application layer (idempotency, persistence, runs)."""


from src.llm_client import AnalyzeResult, LLMRetryableError
from src.models import Inquiry, TokenUsage, compute_inquiry_id
from src.service import InquiryService, estimate_cost
from src.storage import InquiryStore


class RecordingClient:
    """Detailed-capable fake: counts every LLM call, returns a fixed verdict."""

    def __init__(self, usage: TokenUsage | None = None):
        self.calls: list[tuple[str, str]] = []
        self._usage = usage

    def analyze_detailed(self, customer_name: str, message: str) -> AnalyzeResult:
        self.calls.append((customer_name, message))
        return AnalyzeResult(
            payload={"summary": "ok", "category": "General Question", "priority": "Low"},
            attempts=1,
            latency_ms=1.0,
            usage=self._usage,
        )


class FailingOnDemandClient:
    """Fails for messages containing 'fail'; otherwise returns a fixed verdict."""

    def __init__(self):
        self.calls = 0

    def analyze_detailed(self, customer_name: str, message: str) -> AnalyzeResult:
        self.calls += 1
        if "fail" in message.lower():
            raise LLMRetryableError("RateLimitError: slow down", attempts=3)
        return AnalyzeResult(
            payload={"summary": "ok", "category": "Sales", "priority": "Low"},
            attempts=1, latency_ms=2.0,
        )


USAGE = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)


def build_service(store: InquiryStore, client) -> InquiryService:
    return InquiryService(
        llm_client=client, store=store, provider="mock", model="mock-1",
        price_input_per_mtok=0.15, price_output_per_mtok=0.6,
    )


class TestSubmit:
    def test_submit_persists_success_with_usage_and_cost(self, store):
        client = RecordingClient(usage=USAGE)
        outcome = build_service(store, client).submit("Alice", "hi")

        assert outcome.duplicate is False
        record = outcome.record
        assert record.id == compute_inquiry_id("Alice", "hi")
        assert record.status == "success"
        assert record.category == "General Question"
        assert record.provider == "mock"
        assert record.model == "mock-1"
        assert record.total_tokens == 1500
        assert record.estimated_cost == round((1000 * 0.15 + 500 * 0.6) / 1_000_000, 6)

    def test_duplicate_submission_skips_llm_and_returns_existing(self, store):
        client = RecordingClient()
        service = build_service(store, client)

        first = service.submit("Alice", "hi")
        second = service.submit("Alice", "hi")

        assert client.calls == [("Alice", "hi")]  # exactly one LLM call
        assert second.duplicate is True
        assert second.record.id == first.record.id
        assert second.record.summary == first.record.summary

    def test_error_is_persisted_and_not_retried_on_duplicate(self, store):
        client = FailingOnDemandClient()
        service = build_service(store, client)

        failed = service.submit("Alice", "this will fail")
        assert failed.record.status == "error"
        assert "RateLimitError" in failed.record.error_message
        assert failed.record.attempts == 3

        again = service.submit("Alice", "this will fail")
        assert again.duplicate is True
        assert again.record.status == "error"
        assert client.calls == 1  # no second attempt for the same content


class TestBatch:
    def test_batch_processes_everything_and_records_run(self, store):
        client = RecordingClient()
        service = build_service(store, client)
        inquiries = [Inquiry(2, "A", "one"), Inquiry(3, "B", "two"), Inquiry(4, "C", "three")]

        batch = service.process_batch(inquiries)

        assert batch.total == 3
        assert batch.succeeded == 3
        assert batch.failed == 0
        assert batch.duplicates == 0
        assert len(batch.run_id) == 32
        assert len(client.calls) == 3

        run = store.get_run(batch.run_id)
        assert (run.total, run.succeeded, run.failed) == (3, 3, 0)
        assert len(store.inquiries_for_run(batch.run_id)) == 3

    def test_batch_with_duplicates_within_and_across_batches(self, store):
        client = RecordingClient()
        service = build_service(store, client)

        first = service.process_batch([Inquiry(2, "A", "same"), Inquiry(3, "B", "same")])
        assert first.duplicates == 0

        # same content as the previous batch -> both are duplicates, no new LLM calls
        calls_before = len(client.calls)
        second = service.process_batch([Inquiry(2, "A", "same"), Inquiry(3, "B", "same")])
        assert second.duplicates == 2
        assert second.succeeded == 2
        assert len(client.calls) == calls_before
        assert second.records[0].id == first.records[0].id

    def test_batch_partial_failure_isolated(self, store):
        client = FailingOnDemandClient()
        service = build_service(store, client)

        batch = service.process_batch(
            [Inquiry(2, "A", "good"), Inquiry(3, "B", "this will fail"), Inquiry(4, "C", "good")]
        )

        assert (batch.total, batch.succeeded, batch.failed) == (3, 2, 1)
        statuses = {record.status for record in batch.records}
        assert statuses == {"success", "error"}
        run = store.get_run(batch.run_id)
        assert (run.succeeded, run.failed) == (2, 1)


class TestEstimateCost:
    def test_computes_from_configured_price_table(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000, total_tokens=2_000_000)
        assert estimate_cost(usage, price_input_per_mtok=0.15, price_output_per_mtok=0.6) == 0.75

    def test_returns_none_without_usage(self):
        assert estimate_cost(None, price_input_per_mtok=0.15, price_output_per_mtok=0.6) is None

    def test_returns_none_without_configured_prices(self):
        usage = TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20)
        assert estimate_cost(usage, price_input_per_mtok=None, price_output_per_mtok=0.6) is None
        assert estimate_cost(usage, price_input_per_mtok=0.15, price_output_per_mtok=None) is None

    def test_returns_none_when_usage_lacks_token_counts(self):
        usage = TokenUsage(input_tokens=None, output_tokens=10, total_tokens=None)
        assert estimate_cost(usage, price_input_per_mtok=0.15, price_output_per_mtok=0.6) is None


class TestGetInquiry:
    def test_roundtrip_via_service(self, service):
        submitted = service.submit("Alice", "hi")
        fetched = service.get_inquiry(submitted.record.id)
        assert fetched is not None
        assert fetched.message == "hi"

    def test_unknown_id_returns_none(self, service):
        assert service.get_inquiry("missing") is None


def test_processor_passthrough_usage_metadata():
    """End-to-end through the processor: attempts/latency/usage land in the result."""
    from src.processor import InquiryProcessor

    client = RecordingClient(usage=USAGE)
    result = InquiryProcessor(client).process_one(Inquiry(2, "Alice", "hi"))
    assert result.attempts == 1
    assert result.latency_ms >= 0
    assert result.usage.total_tokens == 1500
