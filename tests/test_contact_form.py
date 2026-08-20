"""Tests for the contact form validation and persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.contact.contact_form import (
    save_submission,
    validate_submission,
)


VALID = {"name": "Jane Doe", "email": "jane@example.com", "message": "Great pipeline demo!"}
MISSING_MESSAGE = {"name": "Jane Doe", "email": "jane@example.com", "message": ""}
BAD_EMAIL = {"name": "Jane Doe", "email": "not-an-email", "message": "Hello"}


def test_valid_submission_passes_validation() -> None:
    ok, err = validate_submission(**VALID)
    assert ok is True
    assert err is None


@pytest.mark.parametrize(
    "payload",
    [
        MISSING_MESSAGE,
        {"name": "", "email": "a@b.co", "message": "hi"},
        {"name": "A", "email": "", "message": "hi"},
        {"name": "   ", "email": "a@b.co", "message": "hi"},  # whitespace-only
    ],
)
def test_missing_fields_are_rejected(payload: dict) -> None:
    ok, err = validate_submission(**payload)
    assert ok is False
    assert err == "Please fill in all fields."


@pytest.mark.parametrize(
    "email",
    ["not-an-email", "foo@", "@bar.com", "foo@bar", "foo bar@baz.com"],
)
def test_invalid_email_is_rejected(email: str) -> None:
    ok, err = validate_submission("Jane", email, "hello")
    assert ok is False
    assert err == "Please enter a valid email address."


def test_bad_email_scenario_from_story() -> None:
    ok, err = validate_submission(**BAD_EMAIL)
    assert ok is False
    assert err == "Please enter a valid email address."


def test_save_submission_appends_json_line(tmp_path: Path) -> None:
    log = tmp_path / "contacts.log"
    assert save_submission(**VALID, log_path=log) is True
    assert save_submission(name="Bob", email="b@x.io", message="Yo", log_path=log) is True

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["name"] == "Jane Doe"
    assert first["email"] == "jane@example.com"
    assert first["message"] == "Great pipeline demo!"
    assert "timestamp" in first and first["timestamp"].endswith("+00:00")


def test_save_submission_creates_parent_dir(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "dir" / "contacts.log"
    assert save_submission(**VALID, log_path=log) is True
    assert log.exists()


def test_save_submission_strips_whitespace(tmp_path: Path) -> None:
    log = tmp_path / "contacts.log"
    assert save_submission(
        name="  Jane  ", email="  jane@example.com  ", message="  hi  ", log_path=log
    ) is True
    payload = json.loads(log.read_text(encoding="utf-8").strip())
    assert payload["name"] == "Jane"
    assert payload["email"] == "jane@example.com"
    assert payload["message"] == "hi"


def test_save_submission_returns_false_on_oserror(tmp_path: Path, monkeypatch) -> None:
    log = tmp_path / "contacts.log"

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", _boom)
    assert save_submission(**VALID, log_path=log) is False
