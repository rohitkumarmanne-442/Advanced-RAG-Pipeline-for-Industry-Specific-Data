"""Tests for the contact form validation and logging helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.contact.contact_form import (
    DEFAULT_LOG_PATH,
    _is_valid_email,
    log_contact,
    validate_submission,
)


# ─── validate_submission ────────────────────────────────────────────────

def test_valid_submission_passes():
    ok, level, err = validate_submission("Jane Doe", "jane@example.com", "Great pipeline demo!")
    assert ok is True
    assert level is None
    assert err is None


@pytest.mark.parametrize(
    "name,email,message",
    [
        ("Jane Doe", "jane@example.com", ""),
        ("", "jane@example.com", "hi"),
        ("Jane", "", "hi"),
        ("   ", "jane@example.com", "hi"),  # whitespace-only counts as empty
        ("Jane", "jane@example.com", "   "),
    ],
)
def test_missing_field_returns_warning(name, email, message):
    ok, level, err = validate_submission(name, email, message)
    assert ok is False
    assert level == "warning"
    assert err == "Please fill in all fields."


@pytest.mark.parametrize(
    "bad_email",
    [
        "not-an-email",
        "foo@",
        "@example.com",
        "foo@bar",
        "foo bar@example.com",
        "foo@example.",
    ],
)
def test_invalid_email_returns_error(bad_email):
    ok, level, err = validate_submission("Jane Doe", bad_email, "Hello")
    assert ok is False
    assert level == "error"
    assert err == "Please enter a valid email address."


@pytest.mark.parametrize(
    "good_email",
    [
        "jane@example.com",
        "jane.doe+filter@sub.example.co.uk",
        "a_b-c@d-e.io",
    ],
)
def test_is_valid_email_accepts_reasonable_addresses(good_email):
    assert _is_valid_email(good_email) is True


# ─── log_contact ────────────────────────────────────────────────────────

def test_log_contact_appends_jsonline(tmp_path: Path):
    log_path = tmp_path / "contacts.log"
    log_contact("Jane Doe", "jane@example.com", "Great pipeline demo!", log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["name"] == "Jane Doe"
    assert record["email"] == "jane@example.com"
    assert record["message"] == "Great pipeline demo!"
    assert "timestamp" in record and record["timestamp"]


def test_log_contact_appends_multiple(tmp_path: Path):
    log_path = tmp_path / "contacts.log"
    log_contact("A", "a@example.com", "one", log_path=log_path)
    log_contact("B", "b@example.com", "two", log_path=log_path)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["name"] == "A"
    assert json.loads(lines[1])["name"] == "B"


def test_log_contact_creates_parent_directories(tmp_path: Path):
    log_path = tmp_path / "nested" / "dir" / "contacts.log"
    log_contact("Jane", "jane@example.com", "hi", log_path=log_path)
    assert log_path.exists()


def test_log_contact_strips_whitespace(tmp_path: Path):
    log_path = tmp_path / "contacts.log"
    log_contact("  Jane  ", "  jane@example.com  ", "  hello  ", log_path=log_path)
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["name"] == "Jane"
    assert record["email"] == "jane@example.com"
    assert record["message"] == "hello"


def test_default_log_path_is_repo_relative():
    assert DEFAULT_LOG_PATH == Path("contacts.log")
