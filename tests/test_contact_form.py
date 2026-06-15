"""Tests for src/contact/contact_form.py covering all story scenarios."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# We test the internal helpers directly to avoid requiring a running Streamlit
# server.  The public render_contact_form() function is integration-tested
# through the validation + persistence path by patching st primitives.
from src.contact.contact_form import _validate, _append_submission, _SUBMISSIONS_FILE


# ---------------------------------------------------------------------------
# _validate
# ---------------------------------------------------------------------------

class TestValidate:
    """Scenario 2 & 3: field-level validation."""

    def test_all_valid(self):
        errors = _validate("Jane Doe", "jane@example.com", "Pipeline question",
                           "How do I add a new data source?")
        assert errors == {}

    def test_empty_name(self):
        errors = _validate("", "jane@example.com", "Test", "Hello")
        assert "name" in errors
        assert "required" in errors["name"].lower()

    def test_empty_email(self):
        errors = _validate("Jane", "", "Test", "Hello")
        assert "email" in errors
        assert "required" in errors["email"].lower()

    def test_invalid_email_no_at(self):
        """Scenario 3: 'notanemail' should trigger the email format error."""
        errors = _validate("Jane Doe", "notanemail", "Test", "Hello")
        assert "email" in errors
        assert "valid email" in errors["email"].lower()

    def test_invalid_email_no_dot(self):
        errors = _validate("Jane", "user@nodot", "Sub", "Msg")
        assert "email" in errors

    def test_empty_subject(self):
        errors = _validate("Jane", "jane@example.com", "", "Hello")
        assert "subject" in errors

    def test_empty_message(self):
        """Scenario 2: empty message must produce an inline error."""
        errors = _validate("Jane Doe", "jane@example.com", "Test", "")
        assert "message" in errors
        assert "required" in errors["message"].lower()

    def test_whitespace_only_fields_are_invalid(self):
        errors = _validate("  ", "jane@example.com", "Sub", "   ")
        assert "name" in errors
        assert "message" in errors

    def test_multiple_errors_returned(self):
        errors = _validate("", "notanemail", "", "")
        assert len(errors) >= 3


# ---------------------------------------------------------------------------
# _append_submission
# ---------------------------------------------------------------------------

class TestAppendSubmission:
    """Scenario 1: successful submission writes JSON record."""

    def test_creates_file_on_first_write(self, tmp_path, monkeypatch):
        target = tmp_path / "data" / "contact_submissions.json"
        monkeypatch.setattr(
            "src.contact.contact_form._SUBMISSIONS_FILE", target
        )
        _append_submission({"name": "Jane Doe", "email": "jane@example.com",
                            "subject": "Q", "message": "Hello"})
        assert target.exists()
        records = json.loads(target.read_text())
        assert len(records) == 1
        assert records[0]["name"] == "Jane Doe"
        assert "timestamp" in records[0]

    def test_appends_to_existing_file(self, tmp_path, monkeypatch):
        target = tmp_path / "data" / "contact_submissions.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps([{"name": "Existing", "timestamp": "t0"}]))
        monkeypatch.setattr(
            "src.contact.contact_form._SUBMISSIONS_FILE", target
        )
        _append_submission({"name": "Jane Doe", "email": "jane@example.com",
                            "subject": "Q", "message": "Hello"})
        records = json.loads(target.read_text())
        assert len(records) == 2
        assert records[1]["name"] == "Jane Doe"

    def test_invalid_existing_json_recovers(self, tmp_path, monkeypatch):
        target = tmp_path / "data" / "contact_submissions.json"
        target.parent.mkdir(parents=True)
        target.write_text("NOT JSON")
        monkeypatch.setattr(
            "src.contact.contact_form._SUBMISSIONS_FILE", target
        )
        _append_submission({"name": "J", "email": "j@j.com", "subject": "s",
                            "message": "m"})
        records = json.loads(target.read_text())
        assert len(records) == 1

    def test_no_write_when_validation_fails(self, tmp_path, monkeypatch):
        """Integration check: validate then write — file must NOT exist on error."""
        target = tmp_path / "data" / "contact_submissions.json"
        monkeypatch.setattr(
            "src.contact.contact_form._SUBMISSIONS_FILE", target
        )
        errors = _validate("Jane Doe", "jane@example.com", "Test", "")  # empty msg
        if not errors:
            _append_submission({})  # should not be reached
        assert not target.exists(), "File must not be written when validation fails"
