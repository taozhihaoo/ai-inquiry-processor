"""Command-line entry point.

Works without any API key for ``--help``, ``--version`` and ``--demo`` (a
fully offline mock provider), and reads ``OPENAI_API_KEY`` from the
environment for real runs — never from the command line.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src import __version__
from src.config import AppConfig, load_config
from src.csv_loader import CSVFormatError, load_inquiries
from src.llm_client import MockLLMClient, OpenAIClient
from src.processor import InquiryProcessor
from src.report_writer import summarize_results, write_csv_report, write_json_report

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_CONFIG_ERROR = 2

PROVIDERS = ("openai", "mock")
REPORT_FORMATS = ("json", "csv", "both")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-inquiry-processor",
        description=(
            "Summarize, categorize and prioritize customer inquiries from a CSV "
            "using LLM structured outputs, and emit JSON + CSV reports."
        ),
        epilog=(
            "examples:\n"
            "  python -m src.main --demo                    # offline demo, no API key needed\n"
            "  python -m src.main -i inquiries.csv          # real run via OPENAI_API_KEY\n"
            "  python -m src.main --format csv -o reports   # CSV report into ./reports\n"
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
        "--format", choices=REPORT_FORMATS, default="both",
        help="report format (default: %(default)s)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="run fully offline with a deterministic mock provider (no API key needed)",
    )
    parser.add_argument(
        "--provider", choices=PROVIDERS, default=None,
        help="LLM provider (default: openai, or mock when --demo is given)",
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


def build_client(config: AppConfig) -> object:
    """Instantiate the configured provider client (provider selection lives here only)."""
    if config.provider == "mock":
        return MockLLMClient()
    return OpenAIClient(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        retry_initial_delay=config.retry_initial_delay,
        retry_max_delay=config.retry_max_delay,
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
    config = load_config(provider=provider, model=args.model, output_dir=args.output_dir)

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

    client = build_client(config)
    processor = InquiryProcessor(client)
    logger.info(
        "processing %d inquiries with provider=%s model=%s",
        len(inquiries), config.provider, config.model,
    )
    results = processor.process(inquiries)

    output_dir = Path(config.output_dir)
    written: list[Path] = []
    if args.format in ("json", "both"):
        written.append(
            write_json_report(
                results, output_dir / "inquiries_report.json",
                provider=config.provider, model=config.model,
            )
        )
    if args.format in ("csv", "both"):
        written.append(write_csv_report(results, output_dir / "inquiries_report.csv"))

    summary = summarize_results(results)
    for path in written:
        print(f"report: {path}")
    print(
        f"processed {summary['total']} inquiries: "
        f"{summary['succeeded']} succeeded, {summary['failed']} failed ({summary['success_rate']})"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
