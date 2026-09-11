"""Smoke tests for the contact form UI module.

We can't drive Streamlit end-to-end here, but we can verify:

* the module imports cleanly (so the ``app.py`` wire-in won't crash),
* it exposes the expected public entry point,
* the underlying validation predicate agrees with the UI's decisions on
  the story's test-data payloads.
"""

import importlib

import pytest

from src.contact.handler import handle_contact_submission, is_valid_email


def test_contact_form_module_imports():
    mod = importlib.import_module("src.ui.contact_form")
    assert hasattr(mod, "render_contact_form")
    assert callable(mod.render_contact_form)


@pytest.mark.parametrize(
    "payload, should_pass",
    [
        ({"name": "Jane Doe", "email": "jane@example.com", "message": "Great pipeline!"}, True),
        ({"name": "", "email": "jane@example.com", "message": "Hello"}, False),
        ({"name": "Jane", "email": "notanemail", "message": "Hello"}, False),
        ({"name": "Jane", "email": "jane@example.com", "message": ""}, False),
    ],
)
def test_ui_validation_matches_handler_contract(payload, should_pass):
    """Every payload the UI would forward must be accepted by the handler,
    and every payload the UI would block must be rejected by it."""
    name = (payload["name"] or "").strip()
    email = (payload["email"] or "").strip()
    message = (payload["message"] or "").strip()

    ui_would_forward = bool(name) and bool(email) and is_valid_email(email) and bool(message)
    assert ui_would_forward is should_pass

    if should_pass:
        assert handle_contact_submission(name, email, message) is None
    else:
        with pytest.raises(ValueError):
            handle_contact_submission(name, email, message)
