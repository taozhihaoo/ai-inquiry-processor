"""Tests for CSV loading and basic input validation."""

import pytest

from src.csv_loader import CSVFormatError, load_inquiries


def write_csv(tmp_path, content: str, encoding: str = "utf-8"):
    path = tmp_path / "inquiries.csv"
    path.write_text(content, encoding=encoding)
    return path


def test_loads_valid_rows(tmp_path):
    path = write_csv(
        tmp_path,
        "customer_name,message\n"
        'Alice,"I cannot log in."\n'
        'Bob,"Do you offer annual billing?"\n',
    )
    inquiries = load_inquiries(path)
    assert len(inquiries) == 2
    assert inquiries[0].row_number == 2
    assert inquiries[0].customer_name == "Alice"
    assert inquiries[1].message == "Do you offer annual billing?"


def test_missing_message_column_raises(tmp_path):
    path = write_csv(tmp_path, "customer_name,note\nAlice,hi\n")
    with pytest.raises(CSVFormatError, match="missing required column\\(s\\): message"):
        load_inquiries(path)


def test_missing_customer_name_column_raises(tmp_path):
    path = write_csv(tmp_path, "message\nhello\n")
    with pytest.raises(CSVFormatError, match="customer_name"):
        load_inquiries(path)


def test_empty_file_raises(tmp_path):
    path = write_csv(tmp_path, "")
    with pytest.raises(CSVFormatError, match="no header row"):
        load_inquiries(path)


def test_blank_lines_are_skipped_with_true_line_numbers(tmp_path):
    path = write_csv(
        tmp_path,
        "customer_name,message\n\nAlice,hi\n\n\nBob,hello there\n",
    )
    inquiries = load_inquiries(path)
    assert [(inquiry.row_number, inquiry.customer_name) for inquiry in inquiries] == [
        (3, "Alice"),
        (6, "Bob"),
    ]


def test_row_with_empty_message_is_skipped(tmp_path):
    path = write_csv(tmp_path, "customer_name,message\nAlice,hi\nCarol,\nBob,hello\n")
    inquiries = load_inquiries(path)
    assert [inquiry.customer_name for inquiry in inquiries] == ["Alice", "Bob"]


def test_row_with_empty_name_is_skipped(tmp_path):
    path = write_csv(tmp_path, "customer_name,message\n,hi\nBob,hello\n")
    inquiries = load_inquiries(path)
    assert [inquiry.customer_name for inquiry in inquiries] == ["Bob"]


def test_extra_columns_are_tolerated(tmp_path):
    path = write_csv(
        tmp_path,
        "customer_name,message,source\nAlice,hi,email\n",
    )
    inquiries = load_inquiries(path)
    assert len(inquiries) == 1
    assert inquiries[0].message == "hi"


def test_utf8_bom_is_tolerated(tmp_path):
    path = write_csv(tmp_path, "customer_name,message\nAlice,hi\n", encoding="utf-8-sig")
    inquiries = load_inquiries(path)
    assert inquiries[0].customer_name == "Alice"


def test_whitespace_is_stripped(tmp_path):
    path = write_csv(tmp_path, "customer_name, message\n  Alice ,  hi there  \n")
    inquiries = load_inquiries(path)
    assert inquiries[0].customer_name == "Alice"
    assert inquiries[0].message == "hi there"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_inquiries(tmp_path / "does_not_exist.csv")
