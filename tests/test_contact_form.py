"""
tests/test_contact_form.py

Unit tests for contact_form.validate_form() and submit_contact_form().
All Streamlit and network calls are mocked so tests run headlessly.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out `streamlit` before importing the module under test so tests
# don't need a running Streamlit server.
# ---------------------------------------------------------------------------
_st_stub = types.ModuleType("streamlit")
_st_stub.secrets = {}  # type: ignore[attr-defined]
_st_stub.warning = MagicMock()  # type: ignore[attr-defined]
_st_stub.success = MagicMock()  # type: ignore[attr-defined]
_st_stub.text_input = MagicMock(return_value="")  # type: ignore[attr-defined]
_st_stub.text_area = MagicMock(return_value="")  # type: ignore[attr-defined]
_st_stub.form = MagicMock()  # type: ignore[attr-defined]
_st_stub.form_submit_button = MagicMock(return_value=False)  # type: ignore[attr-defined]
_st_stub.markdown = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("streamlit", _st_stub)

from contact_form import validate_form, submit_contact_form, _get_secret  # noqa: E402


class TestValidateForm(unittest.TestCase):
    """TC-002, TC-004, TC-008, TC-010 — field validation."""

    # TC-001 / TC-002 — happy path
    def test_valid_submission_returns_no_error(self):
        result = validate_form("Jane Doe", "jane@example.com", "Great pipeline demo!")
        self.assertIsNone(result)

    # TC-004 — empty message
    def test_empty_message_returns_warning(self):
        result = validate_form("Jane Doe", "jane@example.com", "")
        self.assertIsNotNone(result)
        self.assertIn("fill", result.lower())

    # TC-004 — empty name
    def test_empty_name_returns_warning(self):
        result = validate_form("", "jane@example.com", "Hello")
        self.assertIsNotNone(result)

    # TC-004 — empty email
    def test_empty_email_returns_warning(self):
        result = validate_form("Jane", "", "Hello")
        self.assertIsNotNone(result)

    # TC-004 — all empty
    def test_all_empty_returns_warning(self):
        result = validate_form("", "", "")
        self.assertIsNotNone(result)

    # TC-010 — whitespace-only fields treated as empty
    def test_whitespace_name_treated_as_empty(self):
        result = validate_form("   ", "jane@example.com", "Hello")
        self.assertIsNotNone(result)
        self.assertIn("fill", result.lower())

    def test_whitespace_message_treated_as_empty(self):
        result = validate_form("Jane", "jane@example.com", "   ")
        self.assertIsNotNone(result)

    def test_whitespace_email_treated_as_empty(self):
        result = validate_form("Jane", "   ", "Hello")
        self.assertIsNotNone(result)

    # TC-008 — malformed email (no @)
    def test_email_without_at_is_invalid(self):
        result = validate_form("John", "notanemail", "Hello")
        self.assertIsNotNone(result)
        self.assertIn("email", result.lower())

    # TC-008 — malformed email (no domain)
    def test_email_without_domain_is_invalid(self):
        result = validate_form("John", "john@", "Hello")
        self.assertIsNotNone(result)
        self.assertIn("email", result.lower())

    def test_email_with_no_tld_is_invalid(self):
        result = validate_form("John", "john@nodomain", "Hello")
        self.assertIsNotNone(result)


class TestSubmitContactForm(unittest.TestCase):
    """Submission routing: Formspree → SMTP → log fallback."""

    def _patch_secret(self, mapping: dict):
        """Return a context manager that mocks _get_secret."""
        return patch(
            "contact_form._get_secret",
            side_effect=lambda k: mapping.get(k),
        )

    # TC-001 / Scenario 4 — no secrets configured → log-only fallback, no crash
    def test_no_secrets_logs_and_does_not_raise(self):
        with self._patch_secret({}):
            with self.assertLogs("contact_form", level="INFO") as cm:
                submit_contact_form("Jane Doe", "jane@example.com", "Great demo!")
        self.assertTrue(any("log-only" in line for line in cm.output))

    # Formspree happy path
    def test_formspree_called_when_url_configured(self):
        with self._patch_secret({"FORMSPREE_URL": "https://formspree.io/f/test"}):
            with patch("contact_form._send_via_formspree") as mock_fs:
                submit_contact_form("Jane", "jane@example.com", "Hi")
                mock_fs.assert_called_once()

    # Formspree failure falls through to SMTP
    def test_formspree_failure_falls_through_to_smtp(self):
        secrets = {
            "FORMSPREE_URL": "https://formspree.io/f/test",
            "SMTP_HOST": "smtp.example.com",
            "SMTP_USER": "sender@example.com",
            "SMTP_PASSWORD": "secret",
            "CONTACT_EMAIL": "rcpt@example.com",
        }
        with self._patch_secret(secrets):
            with patch("contact_form._send_via_formspree", side_effect=RuntimeError("fail")):
                with patch("contact_form._send_via_smtp") as mock_smtp:
                    submit_contact_form("Jane", "jane@example.com", "Hi")
                    mock_smtp.assert_called_once()

    # SMTP happy path
    def test_smtp_called_when_configured(self):
        secrets = {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_USER": "sender@example.com",
            "SMTP_PASSWORD": "secret",
            "CONTACT_EMAIL": "rcpt@example.com",
        }
        with self._patch_secret(secrets):
            with patch("contact_form._send_via_smtp") as mock_smtp:
                submit_contact_form("Jane", "jane@example.com", "Hi")
                mock_smtp.assert_called_once()

    # Whitespace stripped before submission
    def test_whitespace_stripped_before_submission(self):
        captured: list = []

        def fake_send(url, name, email, message):
            captured.extend([name, email, message])

        with self._patch_secret({"FORMSPREE_URL": "https://formspree.io/f/test"}):
            with patch("contact_form._send_via_formspree", side_effect=fake_send):
                submit_contact_form("  Jane  ", "  jane@example.com  ", "  Hi  ")

        self.assertEqual(captured[0], "Jane")
        self.assertEqual(captured[1], "jane@example.com")
        self.assertEqual(captured[2], "Hi")


class TestGetSecret(unittest.TestCase):
    """_get_secret reads st.secrets first, then env vars."""

    def test_env_var_fallback(self):
        import os

        os.environ["TEST_SECRET_XYZ"] = "env_value"
        try:
            val = _get_secret("TEST_SECRET_XYZ")
            self.assertEqual(val, "env_value")
        finally:
            del os.environ["TEST_SECRET_XYZ"]

    def test_missing_secret_returns_none(self):
        val = _get_secret("__NONEXISTENT_SECRET_KEY__")
        self.assertIsNone(val)


if __name__ == "__main__":
    unittest.main()
