"""Tests for src/contact/contact_form.py.

Covers:
  - Happy path: valid data passes validation and is written to JSON
  - Missing required field: per-field errors returned, no file write
  - Invalid email: email-specific error returned, no file write
  - Multiple invalid fields: all errors reported simultaneously
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out streamlit before importing the module under test so tests run
# without a running Streamlit server.
# ---------------------------------------------------------------------------
streamlit_stub = types.ModuleType("streamlit")
for _attr in [
    "markdown", "text_input", "text_area", "form", "form_submit_button",
    "columns", "button", "rerun", "session_state",
]:
    setattr(streamlit_stub, _attr, MagicMock())
streamlit_stub.session_state = {}
sys.modules.setdefault("streamlit", streamlit_stub)

from src.contact.contact_form import _append_submission, _validate  # noqa: E402


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidate:
    """Unit tests for _validate()."""

    VALID = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "subject": "Pipeline question",
        "message": "How do I add a new data source?",
    }

    def test_happy_path_no_errors(self):
        errors = _validate(**self.VALID)
        assert errors == {}

    def test_empty_name_returns_error(self):
        data = {**self.VALID, "name": ""}
        errors = _validate(**data)
        assert "name" in errors
        assert errors["name"] == "Name is required."

    def test_empty_email_returns_error(self):
        data = {**self.VALID, "email": ""}
        errors = _validate(**data)
        assert "email" in errors
        assert errors["email"] == "Email is required."

    def test_invalid_email_no_at_sign(self):
        data = {**self.VALID, "email": "notanemail"}
        errors = _validate(**data)
        assert "email" in errors
        assert errors["email"] == "Please enter a valid email address."

    def test_invalid_email_no_dot(self):
        data = {**self.VALID, "email": "user@nodot"}
        errors = _validate(**data)
        assert "email" in errors

    def test_valid_email_passes(self):
        data = {**self.VALID, "email": "jane@example.com"}
        errors = _validate(**data)
        assert "email" not in errors

    def test_empty_subject_returns_error(self):
        data = {**self.VALID, "subject": ""}
        errors = _validate(**data)
        assert "subject" in errors
        assert errors["subject"] == "Subject is required."

    def test_empty_message_returns_error(self):
        data = {**self.VALID, "message": ""}
        errors = _validate(**data)
        assert "message" in errors
        assert errors["message"] == "Message is required."

    def test_multiple_empty_fields_all_reported(self):
        errors = _validate(name="", email="", subject="", message="")
        assert set(errors.keys()) == {"name", "email", "subject", "message"}

    def test_whitespace_only_treated_as_empty(self):
        data = {**self.VALID, "message": "   "}
        errors = _validate(**data)
        assert "message" in errors


# ---------------------------------------------------------------------------
# Submission persistence tests
# ---------------------------------------------------------------------------

class TestAppendSubmission:
    """Tests for _append_submission() JSON file writing."""

    VALID_DATA = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "subject": "Pipeline question",
        "message": "How do I add a new data source?",
    }

    def test_creates_file_and_appends_record(self, tmp_path):
        submissions_file = tmp_path / "contact_submissions.json"
        with patch("src.contact.contact_form._SUBMISSIONS_FILE", submissions_file):
            _append_submission(**self.VALID_DATA)

        assert submissions_file.exists()
        records = json.loads(submissions_file.read_text())
        assert len(records) == 1
        record = records[0]
        assert record["name"] == "Jane Doe"
        assert record["email"] == "jane@example.com"
        assert record["subject"] == "Pipeline question"
        assert record["message"] == "How do I add a new data source?"
        assert "timestamp" in record

    def test_appends_to_existing_file(self, tmp_path):
        submissions_file = tmp_path / "contact_submissions.json"
        existing = [{"name": "Old Entry", "email": "old@example.com",
                     "subject": "Old", "message": "Old msg",
                     "timestamp": "2024-01-01T00:00:00+00:00"}]
        submissions_file.write_text(json.dumps(existing))

        with patch("src.contact.contact_form._SUBMISSIONS_FILE", submissions_file):
            _append_submission(**self.VALID_DATA)

        records = json.loads(submissions_file.read_text())
        assert len(records) == 2
        assert records[1]["name"] == "Jane Doe"

    def test_handles_corrupted_json_gracefully(self, tmp_path):
        submissions_file = tmp_path / "contact_submissions.json"
        submissions_file.write_text("NOT VALID JSON{{{")

        with patch("src.contact.contact_form._SUBMISSIONS_FILE", submissions_file):
            # Should not raise
            _append_submission(**self.VALID_DATA)

        records = json.loads(submissions_file.read_text())
        assert len(records) == 1

    def test_no_write_when_validation_fails(self, tmp_path):
        """Simulate the caller's responsibility: _append_submission should
        only be called after _validate returns no errors."""
        submissions_file = tmp_path / "contact_submissions.json"
        invalid_data = {**self.VALID_DATA, "message": ""}

        errors = _validate(**invalid_data)
        if not errors:
            with patch("src.contact.contact_form._SUBMISSIONS_FILE", submissions_file):
                _append_submission(**invalid_data)

        # File should NOT exist because validation failed
        assert not submissions_file.exists()
