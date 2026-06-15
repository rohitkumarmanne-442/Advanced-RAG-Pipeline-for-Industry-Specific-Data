"""Tests for ``src.contact_form.validate_contact_form``.

These tests mirror the four Given/When/Then scenarios in the user story
and use the exact test data fixtures supplied by the PM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo root importable when running ``pytest`` from any cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contact_form import (  # noqa: E402  (path setup must come first)
    MIN_MESSAGE_LENGTH,
    SUBJECT_CATEGORIES,
    _is_valid_email,
    validate_contact_form,
)


# ---------------------------------------------------------------------------
# Fixtures from the story's "Test data" block
# ---------------------------------------------------------------------------

VALID = {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "subject": "Bug Report",
    "message": "The pipeline crashes when I upload a PDF larger than 10 MB.",
}

INVALID_EMAIL = {
    "name": "Bob",
    "email": "bob.nodomain",
    "subject": "General Question",
    "message": "Just testing the form submission flow here.",
}

SHORT_MESSAGE = {
    "name": "Alice",
    "email": "alice@test.com",
    "subject": "Feedback",
    "message": "Too short",
}

ALL_EMPTY = {"name": "", "email": "", "subject": "", "message": ""}


# ---------------------------------------------------------------------------
# Sanity check on module constants
# ---------------------------------------------------------------------------

def test_subject_categories_has_at_least_three_real_options():
    # The first entry is the empty placeholder used by the selectbox.
    real_categories = [c for c in SUBJECT_CATEGORIES if c]
    assert len(real_categories) >= 3


def test_min_message_length_is_twenty():
    assert MIN_MESSAGE_LENGTH == 20


# ---------------------------------------------------------------------------
# Scenario 1 — happy path
# ---------------------------------------------------------------------------

def test_valid_submission_has_no_errors():
    errors = validate_contact_form(**VALID)
    assert errors == {}


# ---------------------------------------------------------------------------
# Scenario 2 — empty form blocked
# ---------------------------------------------------------------------------

def test_all_empty_yields_error_per_field():
    errors = validate_contact_form(**ALL_EMPTY)
    assert set(errors.keys()) == {"name", "email", "subject", "message"}
    for field, msg in errors.items():
        assert msg, f"Missing error message for {field!r}"


def test_whitespace_only_fields_are_treated_as_empty():
    errors = validate_contact_form(
        name="   ",
        email="\t",
        subject="",
        message="        ",
    )
    assert set(errors.keys()) == {"name", "email", "subject", "message"}


# ---------------------------------------------------------------------------
# Scenario 3 — invalid email blocked
# ---------------------------------------------------------------------------

def test_invalid_email_blocks_submission():
    errors = validate_contact_form(**INVALID_EMAIL)
    assert "email" in errors
    assert errors["email"] == "Please enter a valid email address"
    # No other fields should error in this scenario.
    assert set(errors.keys()) == {"email"}


@pytest.mark.parametrize(
    "bad_email",
    [
        "john.doe",          # no @
        "john@",              # no domain
        "@example.com",      # no local part
        "john@example",      # no TLD
        "john doe@x.com",    # space in local part
        "john@ex ample.com", # space in domain
    ],
)
def test_is_valid_email_rejects_obviously_bad_addresses(bad_email):
    assert not _is_valid_email(bad_email)


@pytest.mark.parametrize(
    "good_email",
    [
        "jane@example.com",
        "jane.doe+filter@sub.example.co.uk",
        "x_y-z@a-b.io",
    ],
)
def test_is_valid_email_accepts_typical_addresses(good_email):
    assert _is_valid_email(good_email)


# ---------------------------------------------------------------------------
# Scenario 4 — message too short blocked
# ---------------------------------------------------------------------------

def test_short_message_blocks_submission():
    errors = validate_contact_form(**SHORT_MESSAGE)
    assert "message" in errors
    assert errors["message"] == "Message must be at least 20 characters"
    assert set(errors.keys()) == {"message"}


def test_message_exactly_at_minimum_length_is_accepted():
    payload = dict(VALID, message="a" * MIN_MESSAGE_LENGTH)
    assert validate_contact_form(**payload) == {}


def test_message_one_below_minimum_length_is_rejected():
    payload = dict(VALID, message="a" * (MIN_MESSAGE_LENGTH - 1))
    errors = validate_contact_form(**payload)
    assert errors == {"message": "Message must be at least 20 characters"}
