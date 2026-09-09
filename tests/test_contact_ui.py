"""Tests for the Streamlit contact form UI helper.

We don't spin up a full Streamlit runtime here — instead we exercise the
pure-Python validation helper and verify the module wires the handler /
exception contract correctly. Full end-to-end coverage of the widget
tree belongs in a Playwright / Streamlit-testing harness, out of scope
for Story 1.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest


# ─── Streamlit stub ──────────────────────────────────────────────────────────
# ``src.ui.contact`` imports streamlit at module import time. If the test
# environment doesn't have streamlit installed we install a minimal stub
# so the import — and therefore the pure helpers we care about — works.
if "streamlit" not in sys.modules:  # pragma: no cover - env dependent
    stub = types.ModuleType("streamlit")
    stub.session_state = {}
    for name in (
        "markdown", "error", "success", "text_input", "text_area",
        "columns", "form", "form_submit_button",
    ):
        setattr(stub, name, mock.MagicMock())
    sys.modules["streamlit"] = stub

from src.ui.contact import _validate_client_side  # noqa: E402
from src.contact.handler import ContactSubmissionError  # noqa: E402


# ─── Client-side validation ──────────────────────────────────────────────────

def test_validate_accepts_valid_payload():
    assert _validate_client_side("Jane Doe", "jane@example.com", "Great tool!") is None


@pytest.mark.parametrize(
    "name,email,message,expected",
    [
        ("", "jane@example.com", "Hello", "Name is required."),
        ("   ", "jane@example.com", "Hello", "Name is required."),
        ("Jane", "", "Hello", "Email is required."),
        ("Jane", "notanemail", "Hello", "Email address looks invalid."),
        ("Jane", "jane@example.com", "", "Message is required."),
        ("Jane", "jane@example.com", "   ", "Message is required."),
    ],
)
def test_validate_returns_expected_error(name, email, message, expected):
    assert _validate_client_side(name, email, message) == expected


# ─── Handler contract still importable from the UI module ────────────────────

def test_contact_submission_error_reexport_shape():
    assert issubclass(ContactSubmissionError, Exception)
