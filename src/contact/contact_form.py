"""Contact Us form for the RAG Pipeline app.

Provides render_contact_form() which renders a Streamlit form with
client-side validation and appends successful submissions to
data/contact_submissions.json.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

# Path for persisting submissions
_SUBMISSIONS_FILE = Path("data") / "contact_submissions.json"

# Simple email regex: must contain '@' and at least one '.' after it
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate(name: str, email: str, subject: str, message: str) -> Dict[str, str]:
    """Return a dict of field -> error string for any invalid fields."""
    errors: Dict[str, str] = {}
    if not name.strip():
        errors["name"] = "Name is required."
    if not email.strip():
        errors["email"] = "Email is required."
    elif not _EMAIL_RE.match(email.strip()):
        errors["email"] = "Please enter a valid email address."
    if not subject.strip():
        errors["subject"] = "Subject is required."
    if not message.strip():
        errors["message"] = "Message is required."
    return errors


def _append_submission(name: str, email: str, subject: str, message: str) -> None:
    """Append a timestamped record to the JSON submissions file."""
    _SUBMISSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name.strip(),
        "email": email.strip(),
        "subject": subject.strip(),
        "message": message.strip(),
    }

    submissions: list = []
    if _SUBMISSIONS_FILE.exists():
        try:
            with _SUBMISSIONS_FILE.open("r", encoding="utf-8") as fh:
                submissions = json.load(fh)
        except (json.JSONDecodeError, ValueError):
            submissions = []

    submissions.append(record)

    with _SUBMISSIONS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(submissions, fh, indent=2, ensure_ascii=False)


def render_contact_form() -> None:
    """Render the full Contact Us page inside the active Streamlit app."""
    # ---- Page header ----
    st.markdown(
        """
        <div style='
            background: linear-gradient(135deg, var(--primary-color, #667eea) 0%,
                        var(--secondary-color, #764ba2) 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
        '>
            <h1 style='margin:0; font-size:2rem;'>📬 Contact Us</h1>
            <p style='margin:0.5rem 0 0; opacity:0.9;'>
                Have a question, feedback, or found a bug? We'd love to hear from you.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialise session state
    if "contact_submitted" not in st.session_state:
        st.session_state["contact_submitted"] = False
    if "contact_errors" not in st.session_state:
        st.session_state["contact_errors"] = {}

    # ---- Success banner (replaces form after submission) ----
    if st.session_state["contact_submitted"]:
        st.markdown(
            """
            <div style='
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                padding: 1.5rem 2rem;
                border-radius: 12px;
                color: white;
                font-size: 1.1rem;
                font-weight: 600;
                margin-top: 1rem;
            '>
                ✅ Thanks! We\'ll be in touch.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Send another message"):
            st.session_state["contact_submitted"] = False
            st.session_state["contact_errors"] = {}
            st.rerun()
        return

    # ---- Form ----
    errors = st.session_state["contact_errors"]

    with st.form(key="contact_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name *", placeholder="Jane Doe")
        with col2:
            email = st.text_input("Email *", placeholder="jane@example.com")

        subject = st.text_input("Subject *", placeholder="e.g. Pipeline question")
        message = st.text_area(
            "Message *",
            placeholder="How can we help?",
            height=160,
        )

        submitted = st.form_submit_button(
            "Send Message",
            use_container_width=True,
            type="primary",
        )

    # ---- Validation & submission (runs outside the form block) ----
    if submitted:
        new_errors = _validate(name, email, subject, message)
        st.session_state["contact_errors"] = new_errors

        if not new_errors:
            _append_submission(name, email, subject, message)
            st.session_state["contact_submitted"] = True
            st.rerun()
        else:
            st.rerun()  # re-render to show error messages

    # ---- Inline error display (rendered after form so always visible) ----
    if errors:
        for field, msg in errors.items():
            label_map = {
                "name": "Name",
                "email": "Email",
                "subject": "Subject",
                "message": "Message",
            }
            st.markdown(
                f"<p style='color:#e74c3c; margin:0.1rem 0;'>⚠️ <strong>{label_map.get(field, field)}:</strong> {msg}</p>",
                unsafe_allow_html=True,
            )
