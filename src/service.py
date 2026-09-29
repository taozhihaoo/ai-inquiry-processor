"""Application service layer — the single business entry point.

Both the CLI (``src.main``) and the HTTP API (``src.api``) go through
:class:`InquiryService`; neither talks to a provider client or the store
directly. Responsibilities:

- idempotent submission (stable SHA-256 content identity, no duplicate LLM calls)
- processing via :class:`InquiryProcessor` (validation + per-item isolation)
- persistence orchestration (pending -> success/error) and usage recording
- batch runs with a ``processing_runs`` row per batch
- optional cost estimation from a configured price table (never fabricated)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from src.llm_client import LLMClient
from src.models import Inquiry, TokenUsage, compute_inquiry_id
from src.processor import InquiryProcessor
from src.storage import InquiryRecord, InquiryStore, StorageError

logger = logging.getLogger(__name__)


def estimate_cost(
    usage: TokenUsage | None,
    *,
    price_input_per_mtok: float | None,
    price_output_per_mtok: float | None,
) -> float | None:
    """Estimate USD cost from a configured price table (USD per 1M tokens).

    Returns ``None`` — never a fabricated number — when the provider reported
    no usage or either price is not configured.
    """
    if usage is None or price_input_per_mtok is None or price_output_per_mtok is None:
        return None
    if usage.input_tokens is None or usage.output_tokens is None:
        return None
    cost = (
        usage.input_tokens * price_input_per_mtok
        + usage.output_tokens * price_output_per_mtok
    ) / 1_000_000
    return round(cost, 6)


@dataclass(frozen=True)
class SubmitOutcome:
    record: InquiryRecord
    duplicate: bool


@dataclass(frozen=True)
class BatchOutcome:
    run_id: str
    total: int
    succeeded: int
    failed: int
    duplicates: int
    outcomes: list[SubmitOutcome]

    @property
    def records(self) -> list[InquiryRecord]:
        return [outcome.record for outcome in self.outcomes]


class InquiryService:
    """Orchestrates processing, validation, provider invocation and persistence."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        store: InquiryStore,
        provider: str,
        model: str | None = None,
        price_input_per_mtok: float | None = None,
        price_output_per_mtok: float | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._processor = InquiryProcessor(llm_client)
        self._store = store
        self._provider = provider
        self._model = model
        self._price_input_per_mtok = price_input_per_mtok
        self._price_output_per_mtok = price_output_per_mtok

    def submit(self, customer_name: str, message: str, *, run_id: str | None = None) -> SubmitOutcome:
        """Process one inquiry idempotently.

        The first submission for a given content identity is processed and
        persisted; any resubmission returns the stored outcome without calling
        the LLM again and is flagged as a duplicate.
        """
        inquiry_id = compute_inquiry_id(customer_name, message)
        record = InquiryRecord.pending(
            id=inquiry_id,
            customer_name=customer_name,
            message=message,
            provider=self._provider,
            model=self._model,
            run_id=run_id,
        )
        if not self._store.insert_inquiry(record):
            existing = self._store.get_inquiry(inquiry_id)
            if existing is None:  # pragma: no cover - only under concurrent deletion
                raise StorageError(f"inquiry disappeared mid-submit: {inquiry_id}")
            logger.info("duplicate submission for %s — returning stored result", inquiry_id[:12])
            return SubmitOutcome(record=existing, duplicate=True)

        result = self._processor.process_one(
            Inquiry(row_number=0, customer_name=customer_name, message=message)
        )
        if result.is_success:
            self._store.update_success(
                inquiry_id,
                summary=result.analysis.summary,
                category=result.analysis.category,
                priority=result.analysis.priority,
                model=self._model,
                input_tokens=result.usage.input_tokens if result.usage else None,
                output_tokens=result.usage.output_tokens if result.usage else None,
                total_tokens=result.usage.total_tokens if result.usage else None,
                latency_ms=result.latency_ms,
                attempts=result.attempts,
                estimated_cost=estimate_cost(
                    result.usage,
                    price_input_per_mtok=self._price_input_per_mtok,
                    price_output_per_mtok=self._price_output_per_mtok,
                ),
            )
        else:
            self._store.update_error(
                inquiry_id,
                error_message=result.error,
                attempts=result.attempts,
                latency_ms=result.latency_ms,
            )
        return SubmitOutcome(record=self._store.get_inquiry(inquiry_id), duplicate=False)

    def process_batch(self, inquiries: Sequence[Inquiry], *, run_id: str | None = None) -> BatchOutcome:
        """Process a batch as one run; single-item failures never abort the batch."""
        run_id = run_id or uuid.uuid4().hex
        self._store.create_run(run_id, provider=self._provider, model=self._model)
        outcomes = [
            self.submit(inquiry.customer_name, inquiry.message, run_id=run_id)
            for inquiry in inquiries
        ]
        records = [outcome.record for outcome in outcomes]
        succeeded = sum(1 for record in records if record.is_success)
        failed = len(records) - succeeded
        self._store.finish_run(run_id, total=len(records), succeeded=succeeded, failed=failed)
        return BatchOutcome(
            run_id=run_id,
            total=len(records),
            succeeded=succeeded,
            failed=failed,
            duplicates=sum(1 for outcome in outcomes if outcome.duplicate),
            outcomes=outcomes,
        )

    def get_inquiry(self, inquiry_id: str) -> InquiryRecord | None:
        return self._store.get_inquiry(inquiry_id)

    @property
    def store(self) -> InquiryStore:
        """The backing store (exposed for health checks and wiring, not for business logic)."""
        return self._store

    @property
    def provider(self) -> str:
        """Configured provider name (informational, e.g. for /health)."""
        return self._provider

    @property
    def llm_client(self) -> LLMClient:
        """The injected provider client (exposed for wiring composite apps)."""
        return self._llm_client
