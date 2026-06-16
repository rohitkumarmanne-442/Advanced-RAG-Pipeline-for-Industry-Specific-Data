"""Tests for the contact submission handler stub (Story 1)."""

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

def test_handler_accepts_valid_payload():
    # Stub returns None on success; real delivery comes in Story 2.
    assert (
        handle_contact_submission(
            "Jane Doe",
            "jane@example.com",
            "Great pipeline demo! Can we collaborate?",
        )
        is None
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "", "email": "j@e.com", "message": "hi"},
        {"name": "Jane", "email": "", "message": "hi"},
        {"name": "Jane", "email": "notanemail", "message": "hi"},
        {"name": "Jane", "email": "j@e.com", "message": "   "},
    ],
)
def test_handler_rejects_invalid_payloads(payload):
    with pytest.raises(ValueError):
        handle_contact_submission(**payload)


def test_contact_submission_error_is_runtime_error():
    # The UI catches broad Exception, but downstream code may want to
    # distinguish delivery failures specifically.
    assert issubclass(ContactSubmissionError, RuntimeError)
