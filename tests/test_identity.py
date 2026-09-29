"""Tests for the stable inquiry identity (SHA-256 content hashing)."""

import hashlib

from src.models import compute_inquiry_id


def test_id_is_sha256_of_normalized_content():
    expected = hashlib.sha256(b"Alice\nI cannot log in.").hexdigest()
    assert compute_inquiry_id("Alice", "I cannot log in.") == expected


def test_id_is_deterministic():
    assert compute_inquiry_id("Alice", "hi") == compute_inquiry_id("Alice", "hi")


def test_surrounding_whitespace_is_normalized():
    assert compute_inquiry_id("  Alice ", "  hi \t") == compute_inquiry_id("Alice", "hi")


def test_different_names_produce_different_ids():
    assert compute_inquiry_id("Alice", "hi") != compute_inquiry_id("Bob", "hi")


def test_different_messages_produce_different_ids():
    assert compute_inquiry_id("Alice", "hi") != compute_inquiry_id("Alice", "hello")


def test_id_is_not_the_raw_message():
    message = "secret internal text"
    inquiry_id = compute_inquiry_id("Alice", message)
    assert message not in inquiry_id
    assert len(inquiry_id) == 64  # hex-encoded SHA-256


def test_unicode_content_is_supported():
    assert compute_inquiry_id("阿丽", "无法登录") == compute_inquiry_id("阿丽", "无法登录")
    assert compute_inquiry_id("阿丽", "无法登录") != compute_inquiry_id("Alice", "cannot log in")
