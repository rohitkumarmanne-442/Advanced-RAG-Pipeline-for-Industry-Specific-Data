"""Contact form UI (Story 1).

Renders an in-app contact form that matches the existing card / section
header design tokens defined in ``app.py``'s custom CSS. Handles empty
fields, invalid emails and backend failures without wiping user input.

Mirrors the structure of ``src/ui/theme.py``: one public ``render_*``
function that ``app.py`` imports once and calls where the form should
appear.
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from src.contact.handler import (
    ContactSubmissionError,
    handle_contact_submission,
)

# Session-state keys — namespaced so we don't clash with anything else
# in the app. Field values are kept in session_state so that a failed
# backend submission does NOT wipe what the user typed (AC #5).
_NAME_KEY: Final = "contact_form_name"
_EMAIL_KEY: Final = "contact_form_email"
_MESSAGE_KEY: Final = "contact_form_message"
_STATUS_KEY: Final = "contact_form_status"  # dict: {level, text}
_CLEAR_FLAG: Final = "contact_form_clear_on_next_run"


def _inject_styles() -> None:
    """Scoped CSS that layers on top of the app's design tokens.

    Uses the same palette / radii as ``app.py`` (indigo primary, slate
    text, ``--radius-lg`` 16px cards) so the form feels native.
    """
    st.markdown(
        """
        <style>
        .contact-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #6366f1;
            border-radius: 16px;
            padding: 1.5rem 2rem;
            margin: 1rem 0 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .contact-card .stTextInput > label,
        .contact-card .stTextArea  > label {
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            color: #6366f1 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .contact-intro {
            color: #64748b;
            font-size: 0.92rem;
            margin: -0.25rem 0 1rem;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _reset_fields() -> None:
    st.session_state[_NAME_KEY] = ""
    st.session_state[_EMAIL_KEY] = ""
    st.session_state[_MESSAGE_KEY] = ""


def render_contact_form() -> None:
    """Render the contact form. Safe to call once per app run."""
    _inject_styles()

    # If the previous run scheduled a clear (successful submit), do it
    # BEFORE the widgets are instantiated — Streamlit forbids mutating
    # a widget's session-state value after the widget is created.
    if st.session_state.pop(_CLEAR_FLAG, False):
        _reset_fields()

    # Seed defaults so ``st.session_state[...]`` reads are always safe.
    st.session_state.setdefault(_NAME_KEY, "")
    st.session_state.setdefault(_EMAIL_KEY, "")
    st.session_state.setdefault(_MESSAGE_KEY, "")

    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon pink">✉️</div>
            <div class="section-title">Get in Touch</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="contact-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="contact-intro">'
        "Questions, feedback or collaboration ideas? Drop a note below "
        "and I'll get back to you."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form("contact_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.text_input("Name", key=_NAME_KEY, placeholder="Jane Doe")
        with col_b:
            st.text_input(
                "Email", key=_EMAIL_KEY, placeholder="jane@example.com"
            )
        st.text_area(
            "Message",
            key=_MESSAGE_KEY,
            placeholder="How can I help?",
            height=140,
        )
        submitted = st.form_submit_button(
            "📨 Send message", type="primary", use_container_width=False
        )

    if submitted:
        _handle_submit()

    _render_status()

    st.markdown("</div>", unsafe_allow_html=True)


def _handle_submit() -> None:
    """Validate + dispatch. Never raises; writes result to session_state."""
    name = (st.session_state.get(_NAME_KEY) or "").strip()
    email = (st.session_state.get(_EMAIL_KEY) or "").strip()
    message = (st.session_state.get(_MESSAGE_KEY) or "").strip()

    # Client-side validation — mirrors handler contract, but runs first
    # so the backend is not called with obviously bad input (AC #2, #3).
    if not name:
        _set_status("error", "Name is required.")
        return
    if not email:
        _set_status("error", "Email is required.")
        return
    if "@" not in email:
        _set_status("error", "Email address looks invalid.")
        return
    if not message:
        _set_status("error", "Message is required.")
        return

    try:
        handle_contact_submission(name, email, message)
    except ValueError as exc:
        # Handler-level validation (e.g. stricter email rules).
        _set_status("error", str(exc))
        return
    except ContactSubmissionError as exc:
        # Backend / delivery failure — keep field values so user can retry.
        _set_status(
            "error",
            f"Sorry, we couldn't send your message: {exc}. Please try again.",
        )
        return
    except Exception as exc:  # pragma: no cover — defensive catch-all
        _set_status(
            "error",
            f"Unexpected error while sending your message: {exc}.",
        )
        return

    # Success — schedule a field-clear for the NEXT run and show confirmation.
    _set_status(
        "success",
        "✅ Thanks — your message was sent. I'll be in touch soon.",
    )
    st.session_state[_CLEAR_FLAG] = True


def _set_status(level: str, text: str) -> None:
    st.session_state[_STATUS_KEY] = {"level": level, "text": text}


def _render_status() -> None:
    status = st.session_state.get(_STATUS_KEY)
    if not status:
        return
    if status["level"] == "success":
        st.success(status["text"])
    else:
        st.error(status["text"])
