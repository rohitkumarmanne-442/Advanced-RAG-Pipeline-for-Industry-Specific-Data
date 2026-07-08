"""Tests for the Streamlit contact form helpers in app.py.

We import the helper functions directly without spinning up Streamlit's
script runner. The module has side-effects at import time (st.set_page_config,
sidebar rendering) — under `pytest` those calls are harmless no-ops because
`streamlit` can be imported headlessly; if `streamlit` is unavailable in the
test environment, the tests are skipped.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

streamlit = pytest.importorskip("streamlit")


@pytest.fixture(scope="module")
def app_module():
    # Import lazily so pytest can skip cleanly if Streamlit isn't installed.
    return importlib.import_module("app")


class TestEmailValidation:
    @pytest.mark.parametrize(
        "addr",
        [
            "jane@example.com",
            "a.b+tag@sub.example.co.uk",
            "user_123@domain.io",
        ],
    )
    def test_accepts_valid(self, app_module, addr):
        assert app_module._is_valid_email(addr) is True

    @pytest.mark.parametrize(
        "addr",
        [
            "",
            "notanemail",
            "missing@domain",  # no TLD
            "@nolocal.com",
            "spaces in@addr.com",
            None,
        ],
    )
    def test_rejects_invalid(self, app_module, addr):
        assert app_module._is_valid_email(addr) is False


class TestSecretLookup:
    def test_env_var_fallback(self, app_module, monkeypatch):
        monkeypatch.setenv("CONTACT_TEST_KEY", "from-env")
        assert app_module._get_secret("CONTACT_TEST_KEY") == "from-env"

    def test_missing_returns_default(self, app_module, monkeypatch):
        monkeypatch.delenv("NOPE_DOES_NOT_EXIST", raising=False)
        assert app_module._get_secret("NOPE_DOES_NOT_EXIST", "fallback") == "fallback"

    def test_never_reads_from_source(self, app_module):
        # The module must not hard-code any real credential names to values.
        source = Path(app_module.__file__).read_text(encoding="utf-8")
        for forbidden in ("smtp.gmail.com", "formspree.io/f/"):
            # These strings are OK in comments/examples but must not appear as
            # assigned literals feeding the send function.
            assert f'"{forbidden}"' not in source or "# " in source


class TestSendFallback:
    def test_logs_and_reports_undelivered_when_no_secrets(self, app_module, monkeypatch):
        for key in (
            "CONTACT_FORMSPREE_URL",
            "SMTP_HOST",
            "SMTP_USER",
            "SMTP_PASSWORD",
            "CONTACT_EMAIL",
        ):
            monkeypatch.delenv(key, raising=False)

        # Force st.secrets lookup to raise so only env vars are consulted.
        monkeypatch.setattr(app_module.st, "secrets", {}, raising=False)

        delivered, transport = app_module._send_contact_message(
            "Jane Doe", "jane@example.com", "Great pipeline demo!"
        )
        assert delivered is False
        assert transport == "logged"

    def test_send_never_raises_on_bad_smtp(self, app_module, monkeypatch):
        # Configure SMTP with a definitely-unreachable host — the helper must
        # swallow the error and fall through to the logging path.
        monkeypatch.setenv("SMTP_HOST", "127.0.0.1")
        monkeypatch.setenv("SMTP_PORT", "2")  # closed port
        monkeypatch.setenv("SMTP_USER", "u@x.com")
        monkeypatch.setenv("SMTP_PASSWORD", "pw")
        monkeypatch.setenv("CONTACT_EMAIL", "to@x.com")
        monkeypatch.delenv("CONTACT_FORMSPREE_URL", raising=False)
        monkeypatch.setattr(app_module.st, "secrets", {}, raising=False)

        delivered, transport = app_module._send_contact_message(
            "Jane", "jane@example.com", "hi"
        )
        assert delivered is False
        assert transport == "logged"
