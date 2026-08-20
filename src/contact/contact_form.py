"""Contact form component for the Streamlit app.

Provides a self-contained ``render_contact_form`` function that draws a
styled contact card (Name, Email, Message) above the app footer.

On submit the payload is validated and, when valid, appended as one JSON
line per submission to a local ``contacts.log`` file. This keeps the
feature dependency-free (no SMTP/external service) while ensuring
submissions survive a Streamlit rerun.

The validation and persistence helpers are exported separately so they
can be unit-tested without a running Streamlit session.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

try:  # Streamlit is only needed for the rendered widget, not the helpers.
    import streamlit as st
except Exception:  # pragma: no cover - streamlit always present in the app
    st = None  # type: ignore[assignment]


# RFC-5322 is overkill for a contact form; this pragmatic pattern is what
# most production apps use (also matches HTML5 input[type=email] behaviour).
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

DEFAULT_LOG_PATH = Path("contacts.log")

_MISSING_FIELDS_MSG = "Please fill in all fields."
_INVALID_EMAIL_MSG = "Please enter a valid email address."
_SUCCESS_MSG = "Thanks — your message has been sent!"
_WRITE_ERROR_MSG = "Sorry, something went wrong saving your message. Please try again later."


def _is_valid_email(email: str) -> bool:
    """Return True if *email* looks like a valid address."""
    if not email or len(email) > 254:
        return False
    return bool(_EMAIL_RE.match(email.strip()))


def validate_submission(name: str, email: str, message: str) -> Tuple[bool, Optional[str]]:
    """Validate a contact submission.

    Returns ``(is_valid, error_message)``. ``error_message`` is ``None``
    when the submission is valid.
    """
    if not (name or "").strip() or not (email or "").strip() or not (message or "").strip():
        return False, _MISSING_FIELDS_MSG
    if not _is_valid_email(email.strip()):
        return False, _INVALID_EMAIL_MSG
    return True, None


def save_submission(
    name: str,
    email: str,
    message: str,
    log_path: Path = DEFAULT_LOG_PATH,
) -> bool:
    """Append a validated submission to *log_path* as a JSON line.

    Returns ``True`` on success, ``False`` if writing failed. Callers
    should already have validated inputs via :func:`validate_submission`.
    """
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name.strip(),
        "email": email.strip(),
        "message": message.strip(),
    }
    try:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except OSError:
        return False


def _inject_styles() -> None:
    """Inject scoped CSS matching the existing app card design."""
    st.markdown(
        """
        <style>
            .contact-card {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border: 1px solid #e2e8f0;
                border-left: 4px solid #6366f1;
                border-radius: 16px;
                padding: 1.5rem 2rem;
                margin: 1rem 0 1.5rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }
            .contact-card .contact-intro {
                color: #475569;
                font-size: 0.95rem;
                line-height: 1.6;
                margin-bottom: 0.5rem;
            }
            .contact-card [data-baseweb="input"] > div,
            .contact-card [data-baseweb="textarea"] > div {
                border-radius: 12px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_contact_form(log_path: Path = DEFAULT_LOG_PATH) -> None:
    """Render the contact form section on the current Streamlit page.

    Safe to call once near the bottom of ``app.py`` (above the footer).
    Uses ``st.form`` so the form only submits on button click. On a
    successful submission the fields are cleared via ``st.session_state``
    before the widgets are instantiated on the next rerun.
    """
    if st is None:  # pragma: no cover - defensive; streamlit is a hard dep
        return

    _inject_styles()

    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon amber">✉️</div>
            <div class="section-title">Get in Touch</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # If the previous run was a successful submission, reset field state
    # BEFORE the widgets are created (Streamlit forbids mutating a widget
    # key after the widget has been instantiated on the same run).
    if st.session_state.pop("_contact_reset", False):
        for key in ("contact_name", "contact_email", "contact_message"):
            st.session_state[key] = ""

    st.markdown('<div class="contact-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="contact-intro">Have feedback or want to collaborate? '
        "Drop a note below and I'll get back to you.</div>",
        unsafe_allow_html=True,
    )

    with st.form("contact_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            name = st.text_input(
                "Name",
                key="contact_name",
                placeholder="Your name",
                max_chars=120,
            )
        with col_b:
            email = st.text_input(
                "Email",
                key="contact_email",
                placeholder="you@example.com",
                max_chars=254,
            )
        message = st.text_area(
            "Message",
            key="contact_message",
            placeholder="How can I help?",
            height=140,
            max_chars=4000,
        )
        submitted = st.form_submit_button("Send", type="primary", use_container_width=False)

    if submitted:
        is_valid, error = validate_submission(name, email, message)
        if not is_valid:
            if error == _MISSING_FIELDS_MSG:
                st.warning(error)
            else:
                st.error(error)
        elif save_submission(name, email, message, log_path=log_path):
            st.success(_SUCCESS_MSG)
            # Defer field reset until the next rerun (see note above).
            st.session_state["_contact_reset"] = True
        else:
            st.error(_WRITE_ERROR_MSG)

    st.markdown("</div>", unsafe_allow_html=True)
