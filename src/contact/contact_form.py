"""Inline contact form for the Streamlit app.

Provides:
  - render_contact_form(): draw the section header + st.form block, handle
    validation, persist valid submissions, show success/error feedback, and
    reset the input fields on a successful send.
  - save_contact(): append a JSON line to contacts.log (tested directly).
  - is_valid_email(): tiny regex-based email sanity check (tested directly).

The visual style is intentionally aligned with the existing app.py card /
section-header design (same border-radius, palette, and .section-header
markup) so no new CSS classes are required.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st

# RFC-5322 is overkill for a UX check; this pattern rejects the obvious junk
# (missing @, missing TLD, whitespace) which is all criterion #3 asks for.
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

DEFAULT_LOG_PATH = Path("contacts.log")


def is_valid_email(email: str) -> bool:
    """Return True if `email` looks like a plausible address."""
    if not email or not isinstance(email, str):
        return False
    return bool(_EMAIL_RE.match(email.strip()))


def save_contact(
    name: str,
    email: str,
    message: str,
    log_path: Optional[Path] = None,
) -> Path:
    """Append the contact payload to `log_path` as one JSON object per line.

    Returns the resolved log path. Raises OSError on I/O failure so the
    caller can surface a user-facing error.
    """
    path = Path(log_path) if log_path is not None else DEFAULT_LOG_PATH
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name.strip(),
        "email": email.strip(),
        "message": message.strip(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def _reset_form_fields() -> None:
    """Clear the three form input keys in session_state (safe pre-widget)."""
    for key in ("contact_name", "contact_email", "contact_message"):
        if key in st.session_state:
            st.session_state[key] = ""


def render_contact_form(log_path: Optional[Path] = None) -> None:
    """Render the contact section. Call this once, above the footer."""
    # If the previous run flagged a successful send, wipe the widget state
    # BEFORE the widgets are instantiated (Streamlit forbids mutating a
    # widget's session_state entry after the widget has been created).
    if st.session_state.pop("_contact_reset", False):
        _reset_form_fields()

    # Section header — reuses the existing .section-header / .section-icon
    # classes defined in app.py so styling matches the rest of the page.
    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon amber">✉️</div>
            <div class="section-title">Get in Touch</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Wrap the form in the same card look as .query-container / .formula-box
    # (white background, rounded corners, subtle border).
    st.markdown(
        """
        <style>
            .contact-card {
                background: white;
                border: 2px solid #e2e8f0;
                border-radius: 16px;
                padding: 1.5rem 1.75rem;
                margin: 1rem 0 1.5rem;
                transition: border-color 0.2s;
            }
            .contact-card:focus-within { border-color: #6366f1; }
            .contact-intro {
                color: #64748b;
                font-size: 0.92rem;
                margin-bottom: 0.75rem;
                line-height: 1.6;
            }
        </style>
        <div class="contact-card">
            <div class="contact-intro">
                Have a question, spotted a bug, or want to collaborate?
                Send a note and I'll get back to you.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # `clear_on_submit=False` — we manage the reset ourselves so the fields
    # are only cleared on a *valid* submit, not on validation errors.
    with st.form("contact_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            name = st.text_input(
                "Name",
                key="contact_name",
                placeholder="Jane Doe",
                max_chars=120,
            )
        with col_b:
            email = st.text_input(
                "Email",
                key="contact_email",
                placeholder="jane@example.com",
                max_chars=254,
            )
        message = st.text_area(
            "Message",
            key="contact_message",
            placeholder="Tell me what's on your mind…",
            height=140,
            max_chars=4000,
        )
        submitted = st.form_submit_button("Send", type="primary")

    if not submitted:
        return

    # --- Validation ---------------------------------------------------
    if not name.strip() or not email.strip() or not message.strip():
        st.warning("Please fill in all fields.")
        return

    if not is_valid_email(email):
        st.error("Please enter a valid email address.")
        return

    # --- Persist ------------------------------------------------------
    try:
        save_contact(name, email, message, log_path=log_path)
    except OSError as exc:
        st.error(f"Sorry, we couldn't save your message: {exc}")
        return

    st.success("✅ Thanks! Your message has been sent.")
    # Defer field reset to the next run — see top of this function.
    st.session_state["_contact_reset"] = True
    st.rerun()
