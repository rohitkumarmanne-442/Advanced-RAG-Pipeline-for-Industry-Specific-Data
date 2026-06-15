"""Contact form page for the RAG Pipeline Streamlit app."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    """Return a dict of field -> error string for any validation failures."""
    errors: dict[str, str] = {}
    if not name.strip():
        errors["name"] = "Name is required."
    if not email.strip():
        errors["email"] = "Email is required."
    elif not _EMAIL_RE.fullmatch(email.strip()):
        errors["email"] = "Please enter a valid email address."
    if not subject.strip():
        errors["subject"] = "Subject is required."
    if not message.strip():
        errors["message"] = "Message is required."
    return errors


def _append_submission(payload: dict[str, Any]) -> None:
    """Append *payload* as a timestamped JSON record to the submissions file."""
    _SUBMISSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []
    if _SUBMISSIONS_FILE.exists() and _SUBMISSIONS_FILE.stat().st_size > 0:
        try:
            existing = json.loads(_SUBMISSIONS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    existing.append(record)
    _SUBMISSIONS_FILE.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------

def render_contact_form() -> None:
    """Render the Contact Us page inside the current Streamlit app."""
    # ---- Page header (matches app gradient style) -------------------------
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, var(--primary-color, #667eea) 0%,
                        var(--secondary-color, #764ba2) 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        ">
            <h1 style="color: white; margin: 0;">📬 Contact Us</h1>
            <p style="color: rgba(255,255,255,0.85); margin: 0.5rem 0 0;">
                Have a question, feedback, or found a bug? We'd love to hear from you.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Early-exit: show success banner if already submitted -------------
    if st.session_state.get("contact_submitted", False):
        st.success("✅ Thanks! We'll be in touch.", icon="✅")
        if st.button("Send another message"):
            st.session_state["contact_submitted"] = False
            st.rerun()
        return

    # ---- Form -------------------------------------------------------------
    with st.form(key="contact_form", clear_on_submit=False):
        name = st.text_input("Name *", placeholder="Jane Doe")
        email = st.text_input("Email *", placeholder="jane@example.com")
        subject = st.text_input("Subject *", placeholder="Pipeline question")
        message = st.text_area(
            "Message *",
            placeholder="How do I add a new data source?",
            height=160,
        )
        submitted = st.form_submit_button(
            "Send Message",
            use_container_width=True,
            type="primary",
        )

    if not submitted:
        return

    # ---- Validation -------------------------------------------------------
    errors = _validate(name, email, subject, message)

    if errors:
        for field, msg in errors.items():
            st.error(f"**{field.capitalize()}**: {msg}")
        return

    # ---- Persist ----------------------------------------------------------
    _append_submission(
        {
            "name": name.strip(),
            "email": email.strip(),
            "subject": subject.strip(),
            "message": message.strip(),
        }
    )

    st.session_state["contact_submitted"] = True
    st.rerun()
