"""
Contact form component for the Advanced RAG Pipeline Streamlit app.

Provides a self-contained ``render_contact_form`` function that mounts a
``st.form`` block styled to match the existing card / section design in
``app.py`` (border-radius 16px, indigo/purple palette). Submissions are
validated inline and persisted to a local ``contacts.log`` file so they
survive a Streamlit rerun.

The module is deliberately dependency-free (only ``streamlit`` at render
time) so its pure-Python helpers (``validate_submission``, ``log_contact``)
are easy to unit-test without spinning up Streamlit.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

# RFC-5322-lite: good enough for form validation, strict enough to reject
# obvious garbage like "not-an-email". We intentionally avoid the full
# RFC grammar — the goal is UX, not spec compliance.
_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)

DEFAULT_LOG_PATH = Path("contacts.log")

# Field-clearing is implemented by bumping a nonce stored in session_state
# and using it in the widget keys — Streamlit forbids mutating a widget's
# state key directly after it has been instantiated, so we swap the keys.
_NONCE_KEY = "_contact_form_nonce"
_STATUS_KEY = "_contact_form_status"  # ("success"|"warning"|"error", message)


def _is_valid_email(email: str) -> bool:
    """Return True if *email* passes our lightweight format check."""
    if not email or len(email) > 254:
        return False
    return _EMAIL_RE.match(email.strip()) is not None


def validate_submission(name: str, email: str, message: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate a contact-form submission.

    Returns a tuple of ``(is_valid, level, error_message)``. ``level`` is
    ``"warning"`` for missing fields and ``"error"`` for a bad email, so the
    caller can choose the appropriate Streamlit banner.
    """
    if not name or not name.strip() or not email or not email.strip() or not message or not message.strip():
        return False, "warning", "Please fill in all fields."
    if not _is_valid_email(email):
        return False, "error", "Please enter a valid email address."
    return True, None, None


def log_contact(
    name: str,
    email: str,
    message: str,
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """
    Append a JSON-line record of the submission to *log_path*.

    The file is opened in append mode with UTF-8 encoding. Parent directories
    are created if missing. Raises the underlying ``OSError`` on failure so
    the caller can surface a user-facing error.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name.strip(),
        "email": email.strip(),
        "message": message.strip(),
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _current_keys(nonce: int) -> Tuple[str, str, str]:
    return (
        f"contact_name_{nonce}",
        f"contact_email_{nonce}",
        f"contact_message_{nonce}",
    )


def render_contact_form(log_path: Path = DEFAULT_LOG_PATH) -> None:
    """
    Render the contact form inside the current Streamlit page.

    Should be called once, positioned above the page footer. Uses the
    existing ``section-header`` / card CSS classes defined in ``app.py``
    so the visual style matches the rest of the app.
    """
    import streamlit as st  # local import: keep module importable in tests

    if _NONCE_KEY not in st.session_state:
        st.session_state[_NONCE_KEY] = 0

    # Section header — matches other sections in app.py.
    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon pink">✉️</div>
            <div class="section-title">Get in Touch</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scoped styling — reuses the app palette (indigo #6366f1, radius 16px).
    st.markdown(
        """
        <style>
            .contact-card {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 1.5rem 2rem;
                margin: 1rem 0 2rem;
            }
            .contact-card p.contact-intro {
                color: #64748b;
                font-size: 0.92rem;
                margin: 0 0 0.75rem;
            }
            div[data-testid="stForm"] {
                border: none !important;
                padding: 0 !important;
                background: transparent !important;
            }
            div[data-testid="stForm"] .stTextInput input,
            div[data-testid="stForm"] .stTextArea textarea {
                border-radius: 12px !important;
                border: 1px solid #e2e8f0 !important;
            }
            div[data-testid="stForm"] .stTextInput input:focus,
            div[data-testid="stForm"] .stTextArea textarea:focus {
                border-color: #6366f1 !important;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
            }
        </style>
        <div class="contact-card">
            <p class="contact-intro">Questions, feedback, or collaboration ideas? Drop a note below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Surface any status banner from the previous run BEFORE we render the
    # form, so a successful submission's success message is visible even
    # after we bump the nonce to reset the fields.
    status = st.session_state.pop(_STATUS_KEY, None)
    if status:
        level, msg = status
        if level == "success":
            st.success(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.error(msg)

    nonce = st.session_state[_NONCE_KEY]
    name_key, email_key, message_key = _current_keys(nonce)

    with st.form(key=f"contact_form_{nonce}", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            name = st.text_input("Name", key=name_key, placeholder="Your name")
        with col_b:
            email = st.text_input("Email", key=email_key, placeholder="you@example.com")
        message = st.text_area(
            "Message",
            key=message_key,
            placeholder="How can I help?",
            height=140,
        )
        submitted = st.form_submit_button("Send", type="primary", use_container_width=False)

    if not submitted:
        return

    is_valid, level, err = validate_submission(name, email, message)
    if not is_valid:
        # Show inline immediately; do NOT bump the nonce so user input stays.
        if level == "warning":
            st.warning(err)
        else:
            st.error(err)
        return

    try:
        log_contact(name, email, message, log_path=log_path)
    except OSError as exc:
        st.error(f"Could not save your message: {exc}")
        return

    # Success — stash the banner for the next rerun and rotate the widget
    # keys so the fields render empty.
    st.session_state[_STATUS_KEY] = (
        "success",
        "Thanks — your message has been sent! I'll get back to you soon.",
    )
    st.session_state[_NONCE_KEY] = nonce + 1
    st.rerun()
