"""Report generation: structured JSON and flat CSV, one row per inquiry."""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src import __version__
from src.models import InquiryResult

logger = logging.getLogger(__name__)

CSV_FIELDS = ["row_number", "customer_name", "status", "summary", "category", "priority", "error"]
RECORD_CSV_FIELDS = [
    "row_number", "id", "customer_name", "status", "summary", "category", "priority", "error_message",
]


def summarize_results(results: list[InquiryResult]) -> dict:
    """Aggregate success/error counts for reports and CLI output."""
    total = len(results)
    succeeded = sum(1 for result in results if result.is_success)
    failed = total - succeeded
    rate = f"{succeeded / total:.0%}" if total else "n/a"
    return {"total": total, "succeeded": succeeded, "failed": failed, "success_rate": rate}


# -- generic writers -----------------------------------------------------------


def write_csv_rows(rows: list[dict], fieldnames: list[str], path: str | Path) -> Path:
    """Write a list of flat dicts as CSV (values are stringified, missing -> empty)."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})
    logger.info("CSV report written to %s", output_path)
    return output_path


def write_json_document(document: dict, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("JSON report written to %s", output_path)
    return output_path


# -- InquiryResult reports (legacy CSV/JSON shape) -----------------------------


def build_json_report(results: list[InquiryResult], *, provider: str, model: str) -> dict:
    return {
        "report_metadata": {
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "tool": "ai-inquiry-processor",
            "tool_version": __version__,
            "provider": provider,
            "model": model,
        },
        "summary": summarize_results(results),
        "results": [result.to_dict() for result in results],
    }


def write_json_report(
    results: list[InquiryResult], path: str | Path, *, provider: str, model: str
) -> Path:
    report = build_json_report(results, provider=provider, model=model)
    return write_json_document(report, path)


def write_csv_report(results: list[InquiryResult], path: str | Path) -> Path:
    rows = [
        {
            "row_number": result.row_number,
            "customer_name": result.customer_name,
            "status": result.status,
            "summary": result.analysis.summary if result.analysis else "",
            "category": result.analysis.category if result.analysis else "",
            "priority": result.analysis.priority if result.analysis else "",
            "error": result.error or "",
        }
        for result in results
    ]
    return write_csv_rows(rows, CSV_FIELDS, path)


# -- persisted-record reports (service runs) ------------------------------------


def build_run_report(
    rows: list[dict], *, provider: str, model: str | None, run_id: str, database: str | None = None
) -> dict:
    """Assemble the JSON report document for a persisted processing run."""
    total = len(rows)
    succeeded = sum(1 for row in rows if row.get("status") == "success")
    failed = total - succeeded
    duplicates = sum(1 for row in rows if row.get("duplicate"))
    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool": "ai-inquiry-processor",
        "tool_version": __version__,
        "provider": provider,
        "model": model,
        "run_id": run_id,
    }
    if database is not None:
        metadata["database"] = database
    return {
        "report_metadata": metadata,
        "summary": {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "duplicates": duplicates,
            "success_rate": f"{succeeded / total:.0%}" if total else "n/a",
        },
        "results": rows,
    }
