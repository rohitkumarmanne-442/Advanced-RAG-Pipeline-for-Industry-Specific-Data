"""Tests for the Contact Streamlit page (pages/Contact.py).

We avoid spinning up a full Streamlit runtime here — instead we verify:
  * The page file exists at the location Streamlit auto-discovers (`pages/`).
  * The module imports cleanly (syntactically valid, no missing deps at parse).
  * The required SUBJECT_OPTIONS are present.
  * The four form fields and the submit button label appear in the source.
  * Core CSS hooks from app.py are mirrored so the styling stays consistent.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTACT_PAGE = REPO_ROOT / "pages" / "Contact.py"


def test_contact_page_file_exists_in_pages_dir():
    """Streamlit auto-discovers files in `pages/` for sidebar navigation."""
    assert CONTACT_PAGE.is_file(), (
        f"Expected Contact page at {CONTACT_PAGE} so Streamlit shows it in "
        "the sidebar and serves it at /Contact."
    )


def _load_contact_module(monkeypatch) -> types.ModuleType:
    """Import pages/Contact.py with a stubbed `streamlit` module.

    The real streamlit package may not be installed in the test env, and even
    when it is, importing the page executes UI calls. We stub the surface area
    the page touches so we can introspect module-level constants.
    """
    fake_st = types.ModuleType("streamlit")

    def _noop(*_args, **_kwargs):
        return None

    def _passthrough_input(*_args, **_kwargs):
        return ""

    def _selectbox(_label, options=None, index=0, **_kwargs):
        if options:
            return options[index if 0 <= index < len(options) else 0]
        return None

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _form(*_args, **_kwargs):
        return _Ctx()

    def _columns(spec):
        n = spec if isinstance(spec, int) else len(spec)
        return tuple(_Ctx() for _ in range(n))

    fake_st.set_page_config = _noop
    fake_st.markdown = _noop
    fake_st.text_input = _passthrough_input
    fake_st.text_area = _passthrough_input
    fake_st.selectbox = _selectbox
    fake_st.form = _form
    fake_st.form_submit_button = lambda *a, **k: False
    fake_st.columns = _columns
    fake_st.error = _noop
    fake_st.success = _noop
    fake_st.info = _noop
    fake_st.warning = _noop

    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    spec = importlib.util.spec_from_file_location("contact_page", CONTACT_PAGE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_subject_options_include_required_values(monkeypatch):
    module = _load_contact_module(monkeypatch)
    assert hasattr(module, "SUBJECT_OPTIONS"), "SUBJECT_OPTIONS must be defined"
    required = {"General Inquiry", "Bug Report", "Feature Request", "Other"}
    assert required.issubset(set(module.SUBJECT_OPTIONS)), (
        f"Missing required subjects. Got: {module.SUBJECT_OPTIONS}"
    )
    # Story explicitly lists Partnership too.
    assert "Partnership" in module.SUBJECT_OPTIONS


def test_email_regex_validates_basic_cases(monkeypatch):
    module = _load_contact_module(monkeypatch)
    assert module.EMAIL_REGEX.match("jane.doe@example.com")
    assert not module.EMAIL_REGEX.match("not-an-email")
    assert not module.EMAIL_REGEX.match("missing@tld")
    assert not module.EMAIL_REGEX.match("")


@pytest.mark.parametrize(
    "needle",
    [
        "Name",
        "Email",
        "Subject",
        "Message",
        "Send Message",
        # CSS hooks borrowed from app.py to keep the visual language consistent.
        "hero-title",
        "hero-badge",
        "contact-card",
        # Placeholders from the story's test data.
        "Jane Doe",
        "jane.doe@example.com",
    ],
)
def test_contact_page_source_contains_expected_markers(needle):
    source = CONTACT_PAGE.read_text(encoding="utf-8")
    assert needle in source, f"Expected '{needle}' in pages/Contact.py source"
