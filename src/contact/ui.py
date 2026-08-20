"""Streamlit contact form UI.

Renders an ``st.form`` block styled to match the existing app.py card design
(matching border-radius, indigo accent, section-header pattern). Validation
and persistence are delegated to :mod:`src.contact.handler`.
"""

from __future__ import annotations

import streamlit as st

from src.contact.handler import (
    ContactSubmissionError,
    handle_contact_submission,
    is_valid_email,
)

_FORM_KEY = "contact_form"
_FIELD_KEYS = ("contact_name", "contact_email", "contact_message")

_CONTACT_CSS = """
<style>
    .contact-container {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-left: 4px solid #6366f1;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
    }
    .contact-intro {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 0.75rem;
    }
</style>
"""


def _reset_fields() -> None:
    """Clear the form fields after a successful submission."""
    for key in _FIELD_KEYS:
        if key in st.session_state:
            st.session_state[key] = ""


def render_contact_form() -> None:
    """Render the contact form section. Safe to call once per rerun."""
    st.markdown(_CONTACT_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon amber">✉️</div>
            <div class="section-title">Contact the Author</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="contact-container">', unsafe_allow_html=True)
    st.markdown(
        '<div class="contact-intro">Questions, feedback, or collaboration '
        "ideas? Send a note below — all fields are required.</div>",
        unsafe_allow_html=True,
    )

    # If the previous rerun requested a reset, clear before widgets render.
    if st.session_state.pop("_contact_reset", False):
        _reset_fields()

    with st.form(_FORM_KEY, clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            name = st.text_input(
                "Name",
                key="contact_name",
                max_chars=200,
                placeholder="Jane Doe",
            )
        with col_b:
            email = st.text_input(
                "Email",
                key="contact_email",
                max_chars=320,
                placeholder="jane@example.com",
            )
        message = st.text_area(
            "Message",
            key="contact_message",
            max_chars=5000,
            placeholder="What's on your mind?",
            height=140,
        )
        submitted = st.form_submit_button("Send", type="primary")

    if submitted:
        name_s = (name or "").strip()
        email_s = (email or "").strip()
        message_s = (message or "").strip()

        if not name_s or not email_s or not message_s:
            st.warning("Please fill in all fields.")
        elif not is_valid_email(email_s):
            st.error("Please enter a valid email address.")
        else:
            try:
                handle_contact_submission(name_s, email_s, message_s)
            except ValueError as exc:
                st.error(str(exc))
            except ContactSubmissionError as exc:
                st.error(f"Sorry — we couldn't record your message. {exc}")
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Unexpected error: {exc}")
            else:
                st.success(
                    "✅ Thanks! Your message has been recorded — "
                    "I'll get back to you soon."
                )
                # Defer field reset to the next rerun so widget state can
                # be reassigned before the widgets are instantiated.
                st.session_state["_contact_reset"] = True
                _rerun = getattr(st, "rerun", None) or getattr(
                    st, "experimental_rerun", None
                )
                if callable(_rerun):
                    _rerun()

    st.markdown("</div>", unsafe_allow_html=True)
