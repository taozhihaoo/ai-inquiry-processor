"""Tests for batch processing: per-item isolation and failure handling."""

from src.llm_client import LLMRetryableError
from src.models import Inquiry, STATUS_ERROR, STATUS_SUCCESS
from src.processor import InquiryProcessor

VALID_PAYLOAD = {"summary": "ok", "category": "General Question", "priority": "Low"}


class StubClient:
    """Scriptable LLMClient double: payloads by index, optional exceptions."""

    def __init__(self, behaviors):
        self._behaviors = list(behaviors)

    def analyze(self, customer_name: str, message: str) -> dict:
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


def test_all_success(processor, sample_inquiries):
    results = processor.process(sample_inquiries)
    assert len(results) == len(sample_inquiries)
    assert all(result.status == STATUS_SUCCESS for result in results)
    assert all(result.analysis is not None for result in results)


def test_single_failure_does_not_crash_the_batch():
    client = StubClient(
        [VALID_PAYLOAD, LLMRetryableError("RateLimitError: slow down"), VALID_PAYLOAD]
    )
    inquiries = [Inquiry(2, "A", "one"), Inquiry(3, "B", "two"), Inquiry(4, "C", "three")]
    results = InquiryProcessor(client).process(inquiries)

    assert [result.status for result in results] == [
        STATUS_SUCCESS, STATUS_ERROR, STATUS_SUCCESS,
    ]
    failed = results[1]
    assert failed.error == "LLMRetryableError: RateLimitError: slow down"
    assert failed.analysis is None
    assert failed.customer_name == "B"


def test_schema_violation_becomes_row_error():
    client = StubClient([{"summary": "ok", "category": "Nonsense", "priority": "Low"}])
    results = InquiryProcessor(client).process([Inquiry(2, "A", "msg")])
    assert results[0].status == STATUS_ERROR
    assert "invalid category" in results[0].error


def test_unexpected_exception_is_contained():
    client = StubClient([RuntimeError("boom"), VALID_PAYLOAD])
    inquiries = [Inquiry(2, "A", "one"), Inquiry(3, "B", "two")]
    results = InquiryProcessor(client).process(inquiries)

    assert results[0].status == STATUS_ERROR
    assert "RuntimeError: boom" in results[0].error
    assert results[1].status == STATUS_SUCCESS


def test_every_input_yields_a_result(sample_inquiries):
    client = StubClient([LLMRetryableError("down")] * len(sample_inquiries))
    results = InquiryProcessor(client).process(sample_inquiries)
    assert len(results) == len(sample_inquiries)
    assert all(result.status == STATUS_ERROR for result in results)
