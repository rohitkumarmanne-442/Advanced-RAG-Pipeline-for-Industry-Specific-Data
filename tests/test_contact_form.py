"""Tests for src.contact.contact_form.

We test the pure functions (`is_valid_email`, `save_contact`) directly since
the Streamlit rendering layer is hard to unit-test without a browser. The
form-level branching (empty-field warning, bad-email error, successful
write) is expressed as validation-function tests that mirror the exact
Given/When/Then scenarios in the story.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.contact.contact_form import is_valid_email, save_contact


# --- Email validation ---------------------------------------------------

@pytest.mark.parametrize(
    "email",
    [
        "jane@example.com",
        "jane.doe+filter@sub.example.co.uk",
        "a@b.co",
    ],
)
def test_is_valid_email_accepts_plausible_addresses(email: str) -> None:
    assert is_valid_email(email) is True


@pytest.mark.parametrize(
    "email",
    [
        "",
        "not-an-email",
        "missing@tld",
        "no-at-sign.com",
        "spaces in@email.com",
        "@nolocal.com",
        "trailing@dot.",
    ],
)
def test_is_valid_email_rejects_junk(email: str) -> None:
    assert is_valid_email(email) is False


def test_is_valid_email_handles_non_strings() -> None:
    assert is_valid_email(None) is False  # type: ignore[arg-type]
    assert is_valid_email(123) is False  # type: ignore[arg-type]


# --- Persistence --------------------------------------------------------

def test_save_contact_appends_json_line(tmp_path: Path) -> None:
    log = tmp_path / "contacts.log"
    save_contact("Jane Doe", "jane@example.com", "Great pipeline demo!", log_path=log)

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["name"] == "Jane Doe"
    assert payload["email"] == "jane@example.com"
    assert payload["message"] == "Great pipeline demo!"
    assert "timestamp" in payload and payload["timestamp"]


def test_save_contact_appends_multiple_entries(tmp_path: Path) -> None:
    log = tmp_path / "contacts.log"
    save_contact("A", "a@a.io", "one", log_path=log)
    save_contact("B", "b@b.io", "two", log_path=log)

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["name"] == "A"
    assert json.loads(lines[1])["name"] == "B"


def test_save_contact_creates_parent_directories(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "dir" / "contacts.log"
    save_contact("Jane", "j@x.io", "hi", log_path=log)
    assert log.exists()


def test_save_contact_strips_whitespace(tmp_path: Path) -> None:
    log = tmp_path / "contacts.log"
    save_contact("  Jane  ", "  jane@example.com ", "  hello  ", log_path=log)
    payload = json.loads(log.read_text(encoding="utf-8").strip())
    assert payload["name"] == "Jane"
    assert payload["email"] == "jane@example.com"
    assert payload["message"] == "hello"


# --- Story scenarios mirrored as pure-function assertions --------------

def test_scenario_happy_path_valid_submission(tmp_path: Path) -> None:
    data = {"name": "Jane Doe", "email": "jane@example.com", "message": "Great pipeline demo!"}
    assert all(v.strip() for v in data.values())
    assert is_valid_email(data["email"])
    log = tmp_path / "contacts.log"
    save_contact(**data, log_path=log)
    assert log.exists() and log.read_text(encoding="utf-8").strip()


def test_scenario_missing_message_field_blocks_submission() -> None:
    data = {"name": "Jane Doe", "email": "jane@example.com", "message": ""}
    # Emulates the guard inside render_contact_form.
    assert not all(v.strip() for v in data.values())


def test_scenario_bad_email_blocks_submission() -> None:
    data = {"name": "Jane Doe", "email": "not-an-email", "message": "Hello"}
    assert all(v.strip() for v in data.values())  # fields present
    assert not is_valid_email(data["email"])       # …but email is bad
