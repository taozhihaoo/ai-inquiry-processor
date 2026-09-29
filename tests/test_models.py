"""Tests for domain models and strict LLM-response validation."""

import pytest

from src.models import (
    CATEGORIES,
    PRIORITIES,
    RESPONSE_FORMAT,
    Inquiry,
    InquiryAnalysis,
    InquiryResult,
    InvalidLLMResponseError,
)

VALID_PAYLOAD = {
    "summary": "Customer cannot log in after a password reset.",
    "category": "Technical Support",
    "priority": "High",
}


class TestInquiryAnalysis:
    def test_valid_payload_builds_analysis(self):
        analysis = InquiryAnalysis.from_dict(VALID_PAYLOAD)
        assert analysis.summary == VALID_PAYLOAD["summary"]
        assert analysis.category == "Technical Support"
        assert analysis.priority == "High"

    def test_summary_is_stripped(self):
        analysis = InquiryAnalysis.from_dict({**VALID_PAYLOAD, "summary": "  padded  "})
        assert analysis.summary == "padded"

    @pytest.mark.parametrize("missing", ["summary", "category", "priority"])
    def test_missing_field_raises(self, missing):
        payload = {key: value for key, value in VALID_PAYLOAD.items() if key != missing}
        with pytest.raises(InvalidLLMResponseError, match="missing required field"):
            InquiryAnalysis.from_dict(payload)

    def test_invalid_category_raises(self):
        payload = {**VALID_PAYLOAD, "category": "Complaints"}
        with pytest.raises(InvalidLLMResponseError, match="invalid category"):
            InquiryAnalysis.from_dict(payload)

    def test_invalid_priority_raises(self):
        payload = {**VALID_PAYLOAD, "priority": "Critical"}
        with pytest.raises(InvalidLLMResponseError, match="invalid priority"):
            InquiryAnalysis.from_dict(payload)

    @pytest.mark.parametrize("field", ["summary", "category", "priority"])
    def test_non_string_field_raises(self, field):
        payload = {**VALID_PAYLOAD, field: 42}
        with pytest.raises(InvalidLLMResponseError, match="must be a string"):
            InquiryAnalysis.from_dict(payload)

    def test_empty_summary_raises(self):
        payload = {**VALID_PAYLOAD, "summary": "   "}
        with pytest.raises(InvalidLLMResponseError, match="must not be empty"):
            InquiryAnalysis.from_dict(payload)

    def test_non_dict_payload_raises(self):
        with pytest.raises(InvalidLLMResponseError, match="expected a JSON object"):
            InquiryAnalysis.from_dict(["not", "a", "dict"])

    def test_enums_are_the_fixed_contract(self):
        assert CATEGORIES == ("Sales", "Technical Support", "Billing", "General Question")
        assert PRIORITIES == ("Low", "Medium", "High")


class TestSchema:
    def test_response_format_is_strict_json_schema(self):
        schema = RESPONSE_FORMAT["json_schema"]
        assert RESPONSE_FORMAT["type"] == "json_schema"
        assert schema["strict"] is True
        assert schema["schema"]["additionalProperties"] is False

    def test_schema_enums_match_model_constants(self):
        properties = RESPONSE_FORMAT["json_schema"]["schema"]["properties"]
        assert properties["category"]["enum"] == list(CATEGORIES)
        assert properties["priority"]["enum"] == list(PRIORITIES)
        assert RESPONSE_FORMAT["json_schema"]["schema"]["required"] == [
            "summary", "category", "priority",
        ]


class TestInquiryResult:
    def test_ok_result(self):
        analysis = InquiryAnalysis.from_dict(VALID_PAYLOAD)
        result = InquiryResult.ok(Inquiry(3, "Alice", "help"), analysis)
        assert result.is_success
        assert result.analysis is analysis
        assert result.error is None
        assert result.to_dict()["analysis"] == VALID_PAYLOAD

    def test_failed_result(self):
        result = InquiryResult.failed(Inquiry(7, "Bob", "boom"), "LLMRetryableError: down")
        assert not result.is_success
        assert result.analysis is None
        assert result.error == "LLMRetryableError: down"
        assert result.to_dict()["analysis"] is None
