"""CSV input handling with basic validation.

Only structural problems (missing file, missing header columns) raise; rows
that are blank or missing a required value are skipped with a warning so one
bad spreadsheet row cannot poison the batch.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.models import Inquiry

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("customer_name", "message")


class CSVFormatError(ValueError):
    """Raised when the input CSV is structurally invalid."""


def load_inquiries(path: str | Path) -> list[Inquiry]:
    """Load inquiries from a CSV file with ``customer_name,message`` columns.

    - Extra columns are ignored; ``utf-8`` with a leading BOM is tolerated.
    - Blank rows and rows with an empty required value are skipped (warning).
    - ``row_number`` is the 1-based line number in the file, so report readers
      can trace every result back to its source row.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"input CSV not found: {csv_path}")

    inquiries: list[Inquiry] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise CSVFormatError(f"input CSV is empty (no header row): {csv_path}")

        header = [cell.strip() for cell in header]
        missing = [column for column in REQUIRED_COLUMNS if column not in header]
        if missing:
            raise CSVFormatError(
                f"input CSV is missing required column(s): {', '.join(missing)}; found: {header}"
            )
        name_index = header.index("customer_name")
        message_index = header.index("message")

        for line_number, cells in enumerate(reader, start=2):
            if not any(cell.strip() for cell in cells):
                continue  # blank separator line
            name = cells[name_index].strip() if name_index < len(cells) else ""
            message = cells[message_index].strip() if message_index < len(cells) else ""
            if not message:
                logger.warning("skipping line %d: 'message' is empty", line_number)
                continue
            if not name:
                logger.warning("skipping line %d: 'customer_name' is empty", line_number)
                continue
            inquiries.append(Inquiry(row_number=line_number, customer_name=name, message=message))

    logger.info("loaded %d inquiries from %s", len(inquiries), csv_path)
    return inquiries
