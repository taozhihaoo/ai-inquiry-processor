"""Command-line entry point.

Works without any API key for ``--help``, ``--version`` and ``--demo`` (a
fully offline mock provider), and reads ``OPENAI_API_KEY`` from the
environment for real runs — never from the command line.

The CLI contains no business logic: it loads the CSV, hands the inquiries to
the shared :class:`InquiryService` (the same entry point the HTTP API uses),
then renders reports from the persisted records.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src import __version__
from src.config import load_config
from src.csv_loader import CSVFormatError, load_inquiries
from src.llm_client import build_llm_client
from src.report_writer import (
    RECORD_CSV_FIELDS,
    build_run_report,
    write_csv_rows,
    write_json_document,
)
from src.service import InquiryService
from src.storage import InquiryStore

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_CONFIG_ERROR = 2

REPORT_FORMATS = ("json", "csv", "both")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-inquiry-processor",
        description=(
            "Summarize, categorize and prioritize customer inquiries from a CSV "
            "using LLM structured outputs, persist results to SQLite, and emit "
            "JSON + CSV reports."
        ),
        epilog=(
            "examples:\n"
            "  python -m src.main --demo                    # offline demo, no API key needed\n"
            "  python -m src.main -i inquiries.csv          # real run via OPENAI_API_KEY\n"
            "  python -m src.main --demo --db data/demo.db  # explicit SQLite location\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", default="sample_inquiries.csv",
        help="input CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", "-o", default="output",
        help="directory for generated reports (default: %(default)s)",
    )
    parser.add_argument(
        "--db", default=None,
        help="SQLite database path (default: AI_INQUIRY_DB_PATH env var or data/inquiries.db)",
    )
    parser.add_argument(
        "--format", choices=REPORT_FORMATS, default="both",
        help="report format (default: %(default)s)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="run fully offline with a deterministic mock provider (no API key needed)",
    )
    parser.add_argument(
        "--provider", choices=("openai", "mock"), default=None,
        help="LLM provider (default: LLM_PROVIDER env var, or openai; --demo forces mock)",
    )
    parser.add_argument(
        "--model", default=None,
        help="model name (default: OPENAI_MODEL env var or gpt-4o-mini)",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
        help="logging verbosity (default: %(default)s)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def build_service(config):
    """Wire the shared service layer: storage + provider + price configuration."""
    store = InquiryStore(config.db_path)
    store.initialize()
    client = build_llm_client(config)
    return InquiryService(
        llm_client=client,
        store=store,
        provider=config.provider,
        model=config.model,
        price_input_per_mtok=config.price_input_per_mtok,
        price_output_per_mtok=config.price_output_per_mtok,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    provider = "mock" if args.demo else (args.provider or "openai")
    if args.demo:
        logger.info("demo mode: using the deterministic mock provider (no real API calls)")
    config = load_config(
        provider=provider, model=args.model, output_dir=args.output_dir, db_path=args.db
    )

    if config.provider == "openai" and not config.api_key:
        logger.error(
            "OPENAI_API_KEY is not set. Export it or put it in .env (see .env.example), "
            "or run with --demo for an offline mock run."
        )
        return EXIT_CONFIG_ERROR

    try:
        inquiries = load_inquiries(args.input)
    except (FileNotFoundError, CSVFormatError) as exc:
        logger.error("%s", exc)
        return EXIT_INPUT_ERROR

    try:
        service = build_service(config)
    except Exception as exc:  # provider/config wiring problems -> safe CLI error
        logger.error("%s", exc)
        return EXIT_CONFIG_ERROR

    logger.info(
        "processing %d inquiries with provider=%s model=%s db=%s",
        len(inquiries), config.provider, config.model, config.db_path,
    )
    outcome = service.process_batch(inquiries)
    rows = _build_rows(inquiries, outcome)

    written = []
    if args.format in ("json", "both"):
        report = build_run_report(
            rows,
            provider=config.provider,
            model=config.model,
            run_id=outcome.run_id,
            database=str(config.db_path),
        )
        written.append(write_json_document(report, config.output_dir / "inquiries_report.json"))
    if args.format in ("csv", "both"):
        written.append(
            write_csv_rows(rows, RECORD_CSV_FIELDS, config.output_dir / "inquiries_report.csv")
        )

    for path in written:
        print(f"report: {path}")
    print(f"run: {outcome.run_id}")
    print(
        f"processed {outcome.total} inquiries: "
        f"{outcome.succeeded} succeeded, {outcome.failed} failed, "
        f"{outcome.duplicates} duplicates"
    )
    return EXIT_OK


def _build_rows(inquiries, outcome):
    """Pair each input inquiry (with its CSV line number) with its stored record."""
    rows = []
    for inquiry, submit in zip(inquiries, outcome.outcomes):
        record = submit.record
        rows.append(
            {
                "row_number": inquiry.row_number,
                "id": record.id,
                "customer_name": record.customer_name,
                "status": record.status,
                "summary": record.summary,
                "category": record.category,
                "priority": record.priority,
                "error_message": record.error_message,
                "duplicate": submit.duplicate,
            }
        )
    return rows


if __name__ == "__main__":
    sys.exit(main())
