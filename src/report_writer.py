"""Report generation: structured JSON and flat CSV, one row per inquiry."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src import __version__
from src.models import InquiryResult

logger = logging.getLogger(__name__)

CSV_FIELDS = ["row_number", "customer_name", "status", "summary", "category", "priority", "error"]


def summarize_results(results: list[InquiryResult]) -> dict:
    """Aggregate success/error counts for reports and CLI output."""
    total = len(results)
    succeeded = sum(1 for result in results if result.is_success)
    failed = total - succeeded
    rate = f"{succeeded / total:.0%}" if total else "n/a"
    return {"total": total, "succeeded": succeeded, "failed": failed, "success_rate": rate}


def build_json_report(results: list[InquiryResult], *, provider: str, model: str) -> dict:
    return {
        "report_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("JSON report written to %s", output_path)
    return output_path


def write_csv_report(results: list[InquiryResult], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "row_number": result.row_number,
                    "customer_name": result.customer_name,
                    "status": result.status,
                    "summary": result.analysis.summary if result.analysis else "",
                    "category": result.analysis.category if result.analysis else "",
                    "priority": result.analysis.priority if result.analysis else "",
                    "error": result.error or "",
                }
            )
    logger.info("CSV report written to %s", output_path)
    return output_path
