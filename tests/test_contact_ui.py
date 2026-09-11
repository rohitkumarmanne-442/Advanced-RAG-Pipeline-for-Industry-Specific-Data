"""Tests for ``src.ui.contact`` — the Streamlit contact form helper.

Streamlit is heavy and its widgets need a script-run context, so we
unit-test the pure-logic entry point ``_handle_submit`` against a tiny
``streamlit`` stub installed into ``sys.modules`` before import. This
gives us fast, deterministic coverage of every acceptance-criteria
branch (empty field, bad email, success, backend failure).
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict

import pytest


# ─── Streamlit stub ──────────────────────────────────────────────────────────

class _SessionState(dict):
    """dict with attribute access, mimicking ``st.session_state``."""

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        self[name] = value

    def setdefault(self, key, default=None):  # type: ignore[override]
        return super().setdefault(key, default)


def _install_streamlit_stub() -> types.ModuleType:
    if "streamlit" in sys.modules and getattr(
        sys.modules["streamlit"], "__ferb_stub__", False
    ):
        return sys.modules["streamlit"]

    st = types.ModuleType("streamlit")
    st.__ferb_stub__ = True  # type: ignore[attr-defined]
    st.session_state = _SessionState()

    def _noop(*_a, **_k):
        return None

    for name in (
        "markdown", "text_input", "text_area", "success", "error",
        "warning", "info", "caption", "form_submit_button",
    ):
        setattr(st, name, _noop)

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def _form(*_a, **_k):
        return _Ctx()

    def _columns(n, **_k):
        return [_Ctx() for _ in range(n if isinstance(n, int) else len(n))]

    st.form = _form
    st.columns = _columns

    sys.modules["streamlit"] = st
    return st


@pytest.fixture()
def ui(monkeypatch):
    st = _install_streamlit_stub()
    st.session_state.clear()

    # Fresh import so module-level state is reset per test.
    sys.modules.pop("src.ui.contact", None)
    import src.ui.contact as contact_ui  # noqa: WPS433 - deliberate late import

    return contact_ui, st


def _seed(st, name: str, email: str, message: str) -> None:
    st.session_state["contact_form_name"] = name
    st.session_state["contact_form_email"] = email
    st.session_state["contact_form_message"] = message


def _status(st) -> Dict[str, str]:
    return st.session_state.get("contact_form_status") or {}


# ─── Scenarios ───────────────────────────────────────────────────────────────

def test_happy_path_success_clears_fields_on_next_run(ui, monkeypatch):
    contact_ui, st = ui
    calls = []

    def fake_handler(name, email, message):
        calls.append((name, email, message))
        return None

    monkeypatch.setattr(contact_ui, "handle_contact_submission", fake_handler)

    _seed(st, "Jane Doe", "jane@example.com", "Great tool!")
    contact_ui._handle_submit()

    assert calls == [("Jane Doe", "jane@example.com", "Great tool!")]
    assert _status(st)["level"] == "success"
    # Fields are cleared on the NEXT render, not immediately.
    assert st.session_state["contact_form_clear_on_next_run"] is True


def test_empty_name_blocks_backend_call(ui, monkeypatch):
    contact_ui, st = ui
    called = []
    monkeypatch.setattr(
        contact_ui, "handle_contact_submission",
        lambda *a, **k: called.append(a),
    )

    _seed(st, "", "jane@example.com", "Hello")
    contact_ui._handle_submit()

    assert called == []
    assert _status(st) == {"level": "error", "text": "Name is required."}


def test_empty_message_blocks_backend_call(ui, monkeypatch):
    contact_ui, st = ui
    called = []
    monkeypatch.setattr(
        contact_ui, "handle_contact_submission",
        lambda *a, **k: called.append(a),
    )

    _seed(st, "Jane", "jane@example.com", "   ")
    contact_ui._handle_submit()

    assert called == []
    assert _status(st)["level"] == "error"
    assert "Message" in _status(st)["text"]


def test_invalid_email_shows_inline_error(ui, monkeypatch):
    contact_ui, st = ui
    called = []
    monkeypatch.setattr(
        contact_ui, "handle_contact_submission",
        lambda *a, **k: called.append(a),
    )

    _seed(st, "Jane", "notanemail", "Hello")
    contact_ui._handle_submit()

    assert called == []
    assert _status(st) == {
        "level": "error",
        "text": "Email address looks invalid.",
    }


def test_backend_failure_retains_field_values(ui, monkeypatch):
    contact_ui, st = ui

    def boom(*_a, **_k):
        raise contact_ui.ContactSubmissionError("smtp down")

    monkeypatch.setattr(contact_ui, "handle_contact_submission", boom)

    _seed(st, "Jane", "jane@example.com", "Hello")
    contact_ui._handle_submit()

    assert _status(st)["level"] == "error"
    assert "smtp down" in _status(st)["text"]
    # Fields MUST be preserved so the user can retry (AC #5).
    assert st.session_state["contact_form_name"] == "Jane"
    assert st.session_state["contact_form_email"] == "jane@example.com"
    assert st.session_state["contact_form_message"] == "Hello"
    assert "contact_form_clear_on_next_run" not in st.session_state


def test_handler_valueerror_surfaces_as_inline_error(ui, monkeypatch):
    contact_ui, st = ui

    def strict(*_a, **_k):
        raise ValueError("Email address looks invalid.")

    monkeypatch.setattr(contact_ui, "handle_contact_submission", strict)

    # Passes client-side checks (contains '@') but handler rejects it.
    _seed(st, "Jane", "a@b c", "Hello")
    contact_ui._handle_submit()

    assert _status(st) == {
        "level": "error",
        "text": "Email address looks invalid.",
    }
    assert st.session_state["contact_form_email"] == "a@b c"
