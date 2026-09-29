"""Domain data models and strict validation of LLM responses.

Dataclasses are the single modelling choice for this project (no Pydantic /
TypedDict mix). The JSON schema in :data:`ANALYSIS_JSON_SCHEMA` is sent to the
LLM as a *structured output* constraint, and :meth:`InquiryAnalysis.from_dict`
independently re-validates whatever the model actually returned, so a bad
response can never silently enter the pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

CATEGORIES = ("Sales", "Technical Support", "Billing", "General Question")
PRIORITIES = ("Low", "Medium", "High")

STATUS_PENDING = "pending"
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"

ANALYSIS_JSON_SCHEMA: dict = {
    "name": "inquiry_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Concise, neutral one-to-two sentence summary of what the customer needs.",
            },
            "category": {"type": "string", "enum": list(CATEGORIES)},
            "priority": {"type": "string", "enum": list(PRIORITIES)},
        },
        "required": ["summary", "category", "priority"],
        "additionalProperties": False,
    },
}

RESPONSE_FORMAT: dict = {"type": "json_schema", "json_schema": ANALYSIS_JSON_SCHEMA}


class InvalidLLMResponseError(ValueError):
    """Raised when an LLM response does not satisfy the expected schema."""


def compute_inquiry_id(customer_name: str, message: str) -> str:
    """Stable content identity for an inquiry: SHA-256 over normalized fields.

    Two submissions with the same normalized customer name and message map to
    the same id, so persistence layers can deduplicate without calling the
    LLM again. Normalization is strip-only (intentionally conservative).
    """
    normalized = f"{customer_name.strip()}\n{message.strip()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TokenUsage:
    """Token consumption reported by a provider; ``None`` when unavailable."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Inquiry:
    """One input row from the customer-inquiries CSV."""

    row_number: int
    customer_name: str
    message: str


@dataclass(frozen=True)
class InquiryAnalysis:
    """The validated LLM verdict for a single inquiry."""

    summary: str
    category: str
    priority: str

    @classmethod
    def from_dict(cls, data: object) -> "InquiryAnalysis":
        """Validate a raw LLM payload and build an analysis from it.

        Raises:
            InvalidLLMResponseError: if any field is missing, not a string,
                empty, or outside the fixed category/priority enumerations.
        """
        if not isinstance(data, dict):
            raise InvalidLLMResponseError(f"expected a JSON object, got {type(data).__name__}")

        missing = [name for name in ("summary", "category", "priority") if name not in data]
        if missing:
            raise InvalidLLMResponseError(f"missing required field(s): {', '.join(missing)}")

        summary, category, priority = data["summary"], data["category"], data["priority"]

        for name, value in (("summary", summary), ("category", category), ("priority", priority)):
            if not isinstance(value, str):
                raise InvalidLLMResponseError(f"field '{name}' must be a string, got {type(value).__name__}")

        summary = summary.strip()
        if not summary:
            raise InvalidLLMResponseError("field 'summary' must not be empty")
        if category not in CATEGORIES:
            raise InvalidLLMResponseError(f"invalid category {category!r}; expected one of {list(CATEGORIES)}")
        if priority not in PRIORITIES:
            raise InvalidLLMResponseError(f"invalid priority {priority!r}; expected one of {list(PRIORITIES)}")

        return cls(summary=summary, category=category, priority=priority)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InquiryResult:
    """Per-row outcome: every input ends up either ``success`` or ``error``.

    ``usage`` / ``attempts`` / ``latency_ms`` carry provider observability
    metadata; they stay ``None`` when the provider does not supply them.
    """

    row_number: int
    customer_name: str
    message: str
    status: str
    analysis: InquiryAnalysis | None = None
    error: str | None = None
    usage: TokenUsage | None = None
    attempts: int | None = None
    latency_ms: float | None = None

    @classmethod
    def ok(
        cls,
        inquiry: Inquiry,
        analysis: InquiryAnalysis,
        *,
        usage: TokenUsage | None = None,
        attempts: int | None = None,
        latency_ms: float | None = None,
    ) -> "InquiryResult":
        return cls(
            row_number=inquiry.row_number,
            customer_name=inquiry.customer_name,
            message=inquiry.message,
            status=STATUS_SUCCESS,
            analysis=analysis,
            usage=usage,
            attempts=attempts,
            latency_ms=latency_ms,
        )

    @classmethod
    def failed(
        cls,
        inquiry: Inquiry,
        message: str,
        *,
        attempts: int | None = None,
        latency_ms: float | None = None,
    ) -> "InquiryResult":
        return cls(
            row_number=inquiry.row_number,
            customer_name=inquiry.customer_name,
            message=inquiry.message,
            status=STATUS_ERROR,
            error=message,
            attempts=attempts,
            latency_ms=latency_ms,
        )

    @property
    def is_success(self) -> bool:
        return self.status == STATUS_SUCCESS

    def to_dict(self) -> dict:
        return {
            "row_number": self.row_number,
            "customer_name": self.customer_name,
            "message": self.message,
            "status": self.status,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "error": self.error,
        }
