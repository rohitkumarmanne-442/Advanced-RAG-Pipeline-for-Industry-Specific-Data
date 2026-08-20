"""Tests for the contact submission handler."""

import json

import pytest

from src.contact.handler import (
    ContactSubmissionError,
    handle_contact_submission,
    is_valid_email,
)


# ─── Email validation ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "email",
    [
        "jane@example.com",
        "a@b",
        "  jane@example.com  ",  # surrounding whitespace tolerated
        "first.last+tag@sub.domain.co",
    ],
)
def test_is_valid_email_accepts_reasonable_addresses(email):
    assert is_valid_email(email) is True


@pytest.mark.parametrize(
    "email",
    [
        "",
        "notanemail",
        "@nodomain",
        "nolocal@",
        "a b@c.com",
        None,
        123,
    ],
)
def test_is_valid_email_rejects_bad_addresses(email):
    assert is_valid_email(email) is False


# ─── Handler contract ────────────────────────────────────────────────────────

def test_handler_accepts_valid_payload(tmp_path):
    log_file = tmp_path / "contacts.log"
    assert (
        handle_contact_submission(
            "Jane Doe",
            "jane@example.com",
            "Great pipeline demo! Can we collaborate?",
            log_path=log_file,
        )
        is None
    )
    # Persisted as a single JSON line with all three fields.
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["name"] == "Jane Doe"
    assert entry["email"] == "jane@example.com"
    assert entry["message"].startswith("Great pipeline demo!")
    assert "timestamp" in entry


def test_handler_appends_multiple_entries(tmp_path):
    log_file = tmp_path / "contacts.log"
    handle_contact_submission("A", "a@b.co", "one", log_path=log_file)
    handle_contact_submission("B", "b@c.co", "two", log_path=log_file)
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["name"] == "A"
    assert json.loads(lines[1])["name"] == "B"


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "", "email": "j@e.com", "message": "hi"},
        {"name": "Jane", "email": "", "message": "hi"},
        {"name": "Jane", "email": "notanemail", "message": "hi"},
        {"name": "Jane", "email": "j@e.com", "message": "   "},
    ],
)
def test_handler_rejects_invalid_payloads(payload, tmp_path):
    log_file = tmp_path / "contacts.log"
    with pytest.raises(ValueError):
        handle_contact_submission(log_path=log_file, **payload)
    # Nothing should be written for rejected submissions.
    assert not log_file.exists() or log_file.read_text() == ""


def test_contact_submission_error_is_runtime_error():
    # The UI catches broad Exception, but downstream code may want to
    # distinguish delivery failures specifically.
    assert issubclass(ContactSubmissionError, RuntimeError)
