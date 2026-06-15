"""Contact Us page for the RAG Pipeline Streamlit app.

Provides render_contact_form() which displays a validated form and
appends successful submissions to data/contact_submissions.json.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SUBMISSIONS_FILE = Path("data/contact_submissions.json")
_EMAIL_RE = re.compile(r"[^@]+@[^@]+\.[^@]+")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(name: str, email: str, subject: str, message: str) -> dict[str, str]:
    """Return a dict of field -> error message for any failing fields."""
    errors: dict[str, str] = {}
    if not name.strip():
        errors["name"] = "Name is required"
    if not email.strip():
        errors["email"] = "Email is required"
    elif not _EMAIL_RE.fullmatch(email.strip()):
        errors["email"] = "Please enter a valid email address"
    if not subject.strip():
        errors["subject"] = "Subject is required"
    if not message.strip():
        errors["message"] = "Message is required"
    return errors


def _append_submission(record: dict) -> None:
    """Append *record* to the JSON submissions log, creating it if necessary."""
    _SUBMISSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if _SUBMISSIONS_FILE.exists() and _SUBMISSIONS_FILE.stat().st_size > 0:
        try:
            with _SUBMISSIONS_FILE.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except json.JSONDecodeError:
            existing = []

    existing.append(record)

    with _SUBMISSIONS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_contact_form() -> None:
    """Render the Contact Us page inside the current Streamlit app."""

    # ---- Page header (matches app.py gradient style) ----------------------
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, var(--primary-color, #667eea) 0%,
                        var(--secondary-color, #764ba2) 100%);
            padding: 2rem 2.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        ">
            <h1 style="color: white; margin: 0; font-size: 2rem;">📬 Contact Us</h1>
            <p style="color: rgba(255,255,255,0.85); margin: 0.4rem 0 0;">
                Have a question, feedback, or spotted a bug? Drop us a message!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Success state ----------------------------------------------------
    if st.session_state.get("contact_submitted"):
        st.markdown(
            """
            <div style="
                background: #d4edda; border: 1px solid #c3e6cb;
                border-radius: 8px; padding: 1.2rem 1.5rem; color: #155724;
                font-size: 1.05rem;
            ">
                ✅ <strong>Thanks! We'll be in touch.</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Send another message"):
            st.session_state["contact_submitted"] = False
            st.rerun()
        return

    # ---- Form card --------------------------------------------------------
    st.markdown(
        "<div style='border: 1px solid var(--border-color, #e0e0e0); "
        "border-radius: 12px; padding: 2rem;'>",
        unsafe_allow_html=True,
    )

    with st.form(key="contact_form", clear_on_submit=False):
        name_val = st.text_input("Name *", placeholder="Jane Doe")
        email_val = st.text_input("Email *", placeholder="jane@example.com")
        subject_val = st.text_input("Subject *", placeholder="Pipeline question")
        message_val = st.text_area(
            "Message *",
            placeholder="How do I add a new data source?",
            height=150,
        )

        submitted = st.form_submit_button("Send Message", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Validation & persistence ----------------------------------------
    if submitted:
        errors = _validate(name_val, email_val, subject_val, message_val)

        if errors:
            for field, msg in errors.items():
                st.markdown(
                    f"<p style='color:#dc3545; margin: 0.2rem 0 0.6rem;'>"
                    f"⚠️ {msg}</p>",
                    unsafe_allow_html=True,
                )
        else:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "name": name_val.strip(),
                "email": email_val.strip(),
                "subject": subject_val.strip(),
                "message": message_val.strip(),
            }
            _append_submission(record)
            st.session_state["contact_submitted"] = True
            st.rerun()
