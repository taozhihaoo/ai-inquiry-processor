"""Batch processing: exactly one result per inquiry, failures never crash the batch."""

from __future__ import annotations

import logging
from typing import Any

from src.llm_client import AnalyzeResult, LLMClient
from src.models import Inquiry, InquiryAnalysis, InquiryResult

logger = logging.getLogger(__name__)


class InquiryProcessor:
    """Runs every inquiry through an :class:`LLMClient` and collects per-row results.

    The client is injected (dependency injection), so the processor is
    agnostic to the provider and trivially testable with fakes. Providers that
    expose ``analyze_detailed`` also contribute attempts/latency/usage
    metadata; plain ``analyze`` clients keep working unchanged.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def process(self, inquiries: list[Inquiry]) -> list[InquiryResult]:
        return [self.process_one(inquiry) for inquiry in inquiries]

    def process_one(self, inquiry: Inquiry) -> InquiryResult:
        try:
            outcome = self._analyze(inquiry)
            analysis = InquiryAnalysis.from_dict(outcome.payload)
        except Exception as exc:  # broad on purpose: the batch must survive any single failure
            logger.warning(
                "row %d (%s) failed: %s", inquiry.row_number, inquiry.customer_name, exc
            )
            return InquiryResult.failed(
                inquiry,
                f"{type(exc).__name__}: {exc}",
                attempts=getattr(exc, "attempts", None),
            )
        logger.info(
            "row %d (%s): %s / %s",
            inquiry.row_number,
            inquiry.customer_name,
            analysis.category,
            analysis.priority,
        )
        return InquiryResult.ok(
            inquiry,
            analysis,
            usage=outcome.usage,
            attempts=outcome.attempts,
            latency_ms=outcome.latency_ms,
        )

    def _analyze(self, inquiry: Inquiry) -> AnalyzeResult:
        detailed = getattr(self._llm_client, "analyze_detailed", None)
        if callable(detailed):
            return detailed(inquiry.customer_name, inquiry.message)
        payload: dict[str, Any] = self._llm_client.analyze(inquiry.customer_name, inquiry.message)
        return AnalyzeResult(payload=payload)
