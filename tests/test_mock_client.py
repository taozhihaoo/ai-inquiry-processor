"""Tests for the deterministic mock provider used by demo mode."""

from src.llm_client import MockLLMClient

ALICE = "I cannot log into my account after resetting my password."
URGENT = "Our entire team has been locked out since this morning and we cannot process any orders. This is urgent."
SALES = "Could you send me a quote for the Enterprise plan with 50 seats?"
BILLING = "I was charged twice for my September invoice, please refund the duplicate payment."
GENERAL = "How do I export my data to CSV?"
LONG = (
    "This is a very long complaint sentence without any terminal punctuation that just keeps "
    "going and going and going so the summary has to be truncated at some point to remain useful."
)


def test_payload_has_exactly_the_schema_keys():
    payload = MockLLMClient().analyze("Alice", ALICE)
    assert set(payload) == {"summary", "category", "priority"}


def test_technical_support_detection():
    payload = MockLLMClient().analyze("Alice", ALICE)
    assert payload["category"] == "Technical Support"
    assert payload["priority"] == "Medium"


def test_urgent_technical_issue_is_high_priority():
    payload = MockLLMClient().analyze("Carla", URGENT)
    assert payload["category"] == "Technical Support"
    assert payload["priority"] == "High"


def test_sales_detection():
    payload = MockLLMClient().analyze("Dan", SALES)
    assert payload["category"] == "Sales"
    assert payload["priority"] == "Low"


def test_billing_detection():
    payload = MockLLMClient().analyze("Elena", BILLING)
    assert payload["category"] == "Billing"
    assert payload["priority"] == "Medium"


def test_general_question_detection():
    payload = MockLLMClient().analyze("Frank", GENERAL)
    assert payload["category"] == "General Question"
    assert payload["priority"] == "Low"


def test_is_deterministic():
    client = MockLLMClient()
    assert client.analyze("Alice", ALICE) == client.analyze("Alice", ALICE)


def test_summary_contains_name_and_first_sentence():
    payload = MockLLMClient().analyze("Alice", ALICE)
    assert payload["summary"] == f"Alice: {ALICE}"


def test_long_summary_is_truncated():
    payload = MockLLMClient().analyze("Zoe", LONG)
    assert payload["summary"].startswith("Zoe: ")
    assert payload["summary"].endswith("...")
    assert len(payload["summary"]) <= len("Zoe: ") + MockLLMClient.SUMMARY_MAX_CHARS
