"""Tests for JSON/CSV report generation."""

import csv
import json

from src.report_writer import (
    CSV_FIELDS,
    build_json_report,
    summarize_results,
    write_csv_report,
    write_json_report,
)
from src.models import Inquiry, InquiryAnalysis, InquiryResult

ANALYSIS = InquiryAnalysis(summary="Cannot log in.", category="Technical Support", priority="High")


def make_results():
    return [
        InquiryResult.ok(Inquiry(2, "Alice", "Cannot log in."), ANALYSIS),
        InquiryResult.failed(Inquiry(3, "Bob", "Broken app"), "LLMRetryableError: rate limited"),
    ]


class TestSummary:
    def test_counts_and_rate(self):
        summary = summarize_results(make_results())
        assert summary == {"total": 2, "succeeded": 1, "failed": 1, "success_rate": "50%"}

    def test_empty_batch(self):
        assert summarize_results([])["success_rate"] == "n/a"


class TestJsonReport:
    def test_structure(self, tmp_path):
        path = write_json_report(
            make_results(), tmp_path / "report.json", provider="openai", model="gpt-4o-mini"
        )
        report = json.loads(path.read_text(encoding="utf-8"))

        metadata = report["report_metadata"]
        assert metadata["tool"] == "ai-inquiry-processor"
        assert metadata["provider"] == "openai"
        assert metadata["model"] == "gpt-4o-mini"
        assert "generated_at_utc" in metadata

        assert report["summary"]["total"] == 2
        assert len(report["results"]) == 2

        success_row = report["results"][0]
        assert success_row["status"] == "success"
        assert success_row["analysis"]["category"] == "Technical Support"

        error_row = report["results"][1]
        assert error_row["status"] == "error"
        assert error_row["analysis"] is None
        assert "rate limited" in error_row["error"]

    def test_creates_missing_directories(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "report.json"
        path = write_json_report([], nested, provider="mock", model="mock-1")
        assert path.exists()


class TestCsvReport:
    def test_rows_and_columns(self, tmp_path):
        path = write_csv_report(make_results(), tmp_path / "report.csv")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert list(rows[0]) == CSV_FIELDS
        assert rows[0]["status"] == "success"
        assert rows[0]["category"] == "Technical Support"
        assert rows[0]["priority"] == "High"
        assert rows[0]["error"] == ""

        assert rows[1]["status"] == "error"
        assert rows[1]["summary"] == ""
        assert rows[1]["category"] == ""
        assert "rate limited" in rows[1]["error"]

    def test_creates_missing_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "report.csv"
        path = write_csv_report([], nested)
        assert path.exists()
        assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(CSV_FIELDS)
