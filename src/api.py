"""HTTP API layer (FastAPI).

A thin transport wrapper around the shared :class:`InquiryService` — all
business rules (idempotency, validation, persistence, error isolation) live
in the service layer, and the API never touches a provider client or the
database directly.

Error policy: provider/storage failures are converted into safe JSON error
responses — no API keys, no internal tracebacks. The generic ``Exception``
handler deliberately returns an opaque message (details stay in the logs).
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.config import load_config
from src.desk.api import build_desk_router, register_desk_exception_handlers
from src.desk.service import DeskService
from src.desk.storage import DeskStore
from src.llm_client import LLMClientError, LLMRetryableError, build_llm_client
from src.models import Inquiry
from src.service import InquiryService
from src.storage import InquiryRecord, InquiryStore, StorageError

logger = logging.getLogger(__name__)


# -- request / response schemas (API boundary only) -----------------------------


class InquiryIn(BaseModel):
    customer_name: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=10_000)

    @field_validator("customer_name", "message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class BatchIn(BaseModel):
    inquiries: list[InquiryIn] = Field(min_length=1, max_length=1_000)


class AnalysisOut(BaseModel):
    summary: str
    category: str
    priority: str


class InquiryOut(BaseModel):
    id: str
    status: str
    summary: str | None = None
    category: str | None = None
    priority: str | None = None
    error_message: str | None = None
    duplicate: bool = False
    created_at: str | None = None
    updated_at: str | None = None


class BatchItemOut(InquiryOut):
    row_number: int


class BatchOut(BaseModel):
    run_id: str
    total: int
    succeeded: int
    failed: int
    duplicates: int
    results: list[BatchItemOut]


def record_to_out(record: InquiryRecord, *, duplicate: bool = False) -> InquiryOut:
    return InquiryOut(
        id=record.id,
        status=record.status,
        summary=record.summary,
        category=record.category,
        priority=record.priority,
        error_message=record.error_message,
        duplicate=duplicate,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


# -- wiring ---------------------------------------------------------------------


def create_app(
    *,
    service: InquiryService | None = None,
    desk_service: DeskService | None = None,
    provider: str | None = None,
    model: str | None = None,
    db_path: str | None = None,
) -> FastAPI:
    """Build the FastAPI app (inquiry pipeline + support desk).

    ``service`` / ``desk_service`` accept pre-built services (used by tests);
    when omitted both are wired from configuration over the same database.
    """
    app = FastAPI(
        title="AI Inquiry Processor",
        description=(
            "Inquiry triage (LLM structured outputs) and an AI customer support desk: "
            "customers, tickets, tags, assignment, notes, history, search, audit."
        ),
        version="1.0.0",
    )
    state = app.state

    if service is not None:
        state.service = service
        state.storage = service.store
    else:
        # Module-level apps (uvicorn src.api:app) default to the mock provider so the
        # service starts without credentials; set LLM_PROVIDER=openai for real runs.
        import os

        resolved_provider = provider or os.environ.get("LLM_PROVIDER") or "mock"
        config = load_config(provider=resolved_provider, model=model, db_path=db_path)
        store = InquiryStore(config.db_path)
        store.initialize()
        llm_client = build_llm_client(config)
        state.service = InquiryService(
            llm_client=llm_client,
            store=store,
            provider=config.provider,
            model=config.model,
            price_input_per_mtok=config.price_input_per_mtok,
            price_output_per_mtok=config.price_output_per_mtok,
        )
        state.storage = store
        if desk_service is None:
            desk_store = DeskStore(config.db_path)
            desk_store.initialize()
            desk_service = DeskService(
                store=desk_store, llm_client=llm_client,
                provider=config.provider, model=config.model,
            )

    if desk_service is None:
        desk_store = DeskStore(":memory:")
        desk_store.initialize()
        desk_service = DeskService(
            store=desk_store, llm_client=state.service.llm_client,
        )
    state.desk_service = desk_service
    register_desk_exception_handlers(app)
    app.include_router(build_desk_router(desk_service))

    @app.exception_handler(LLMClientError)
    async def _llm_error_handler(_: Request, exc: LLMClientError) -> JSONResponse:
        """Provider failures become a safe 502 — message only, no traceback, no key."""
        logger.warning("provider failure: %s", exc)
        kind = "provider_rate_limited" if isinstance(exc, LLMRetryableError) else "provider_error"
        payload: dict = {"error": {"code": kind, "message": str(exc)}}
        if getattr(exc, "retry_after", None):
            payload["error"]["retry_after_seconds"] = exc.retry_after
        return JSONResponse(status_code=502, content=payload)

    @app.exception_handler(StorageError)
    async def _storage_error_handler(_: Request, exc: StorageError) -> JSONResponse:
        logger.error("storage failure: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "storage_error", "message": "database operation failed"}},
        )

    @app.exception_handler(Exception)
    async def _unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
        """Opaque 500 for anything else — no internals in the response body."""
        logger.exception("unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "unexpected internal error"}},
        )

    @app.get("/health")
    def health() -> dict:
        storage = state.storage
        database_ok = storage.ping() if storage is not None else True
        payload = {"status": "ok" if database_ok else "degraded", "database": "ok" if database_ok else "error"}
        service: InquiryService | None = state.service
        if service is not None:
            payload["provider"] = service.provider
        return payload

    @app.post("/inquiries", status_code=201)
    def submit_inquiry(body: InquiryIn) -> InquiryOut:
        outcome = state.service.submit(body.customer_name, body.message)
        return record_to_out(outcome.record, duplicate=outcome.duplicate)

    @app.post("/inquiries/batch", status_code=200)
    def submit_batch(body: BatchIn) -> BatchOut:
        inquiries = [
            Inquiry(row_number=index, customer_name=item.customer_name, message=item.message)
            for index, item in enumerate(body.inquiries, start=1)
        ]
        batch = state.service.process_batch(inquiries)
        return BatchOut(
            run_id=batch.run_id,
            total=batch.total,
            succeeded=batch.succeeded,
            failed=batch.failed,
            duplicates=batch.duplicates,
            results=[
                BatchItemOut(
                    **record_to_out(outcome.record, duplicate=outcome.duplicate).model_dump(),
                    row_number=index,
                )
                for index, outcome in enumerate(batch.outcomes, start=1)
            ],
        )

    @app.get("/inquiries/{inquiry_id}")
    def get_inquiry(inquiry_id: str) -> InquiryOut:
        record = state.service.get_inquiry(inquiry_id)
        if record is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "not_found", "message": f"inquiry {inquiry_id} not found"}},
            )
        return record_to_out(record)

    return app


app = create_app()
