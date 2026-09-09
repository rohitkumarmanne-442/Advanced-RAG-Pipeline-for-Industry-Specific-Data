"""Contact form UI component for the Streamlit app.

Mirrors the layout convention used by :mod:`src.ui.theme` — a single
``render_*`` function that owns its own markup and Streamlit widgets and
can be dropped into ``app.py`` with a one-line call.

The form collects Name / Email / Message, performs light client-side
validation, and delegates to :func:`src.contact.handler.handle_contact_submission`
for delivery. Validation failures (``ValueError``) are surfaced inline
without wiping the form; delivery failures (``ContactSubmissionError``)
are shown as a red error banner and the entered values are preserved so
the user can retry.
"""

from __future__ import annotations

import streamlit as st

from src.contact.handler import (
    ContactSubmissionError,
    handle_contact_submission,
    is_valid_email,
)

# Keys used for Streamlit's session_state. Centralised so the render
# function and the post-submit "clear on success" logic agree.
_NAME_KEY = "contact_form_name"
_EMAIL_KEY = "contact_form_email"
_MESSAGE_KEY = "contact_form_message"
_SUCCESS_FLAG = "contact_form_just_succeeded"

# Scoped CSS — deliberately reuses the existing design tokens
# (colours, radius, section-header pattern) already defined in app.py so
# the form visually belongs to the same design system.
_CONTACT_CSS = """
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
    .contact-intro {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
</style>
"""


def _validate_client_side(name: str, email: str, message: str) -> str | None:
    """Return an error message, or ``None`` if the payload looks OK.

    Kept in-UI so we can short-circuit before calling the backend handler
    (acceptance criterion #2 explicitly requires the handler NOT be
    invoked for empty fields).
    """
    if not name.strip():
        return "Name is required."
    if not email.strip():
        return "Email is required."
    if not is_valid_email(email):
        return "Email address looks invalid."
    if not message.strip():
        return "Message is required."
    return None


def _clear_fields() -> None:
    for key in (_NAME_KEY, _EMAIL_KEY, _MESSAGE_KEY):
        if key in st.session_state:
            st.session_state[key] = ""


def render_contact_form() -> None:
    """Render the contact form. Safe to call once per page render."""
    st.markdown(_CONTACT_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon pink">✉️</div>
            <div class="section-title">Get in Touch</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # If the previous run submitted successfully, clear the widget state
    # BEFORE the widgets are instantiated (Streamlit forbids mutating a
    # widget's session_state key after it has been created in the same run).
    if st.session_state.pop(_SUCCESS_FLAG, False):
        _clear_fields()

    st.markdown('<div class="contact-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="contact-intro">'
        "Questions, feedback, or collaboration ideas? Drop a note below "
        "and I'll get back to you."
        "</div>",
        unsafe_allow_html=True,
    )

    # ``clear_on_submit=False`` — we clear manually on success only, so
    # that a backend failure preserves the user's input (criterion #5).
    with st.form("contact_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            name = st.text_input(
                "Name",
                key=_NAME_KEY,
                placeholder="Jane Doe",
                max_chars=120,
            )
        with col_b:
            email = st.text_input(
                "Email",
                key=_EMAIL_KEY,
                placeholder="you@example.com",
                max_chars=254,
            )

        message = st.text_area(
            "Message",
            key=_MESSAGE_KEY,
            placeholder="What's on your mind?",
            height=140,
            max_chars=2000,
        )

        submitted = st.form_submit_button("Send message", type="primary")

    if submitted:
        error = _validate_client_side(name, email, message)
        if error is not None:
            st.error(error)
        else:
            try:
                handle_contact_submission(name.strip(), email.strip(), message.strip())
            except ValueError as exc:
                # Backend echoed a validation issue we didn't catch — surface
                # it inline; form values are preserved automatically.
                st.error(str(exc))
            except ContactSubmissionError as exc:
                st.error(f"Sorry, we couldn't deliver your message: {exc}")
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Unexpected error while sending your message: {exc}")
            else:
                st.success(
                    "✅ Thanks for reaching out! Your message has been received."
                )
                # Defer the field-clear to the NEXT run so we don't mutate
                # widget state after it has already been rendered.
                st.session_state[_SUCCESS_FLAG] = True

    st.markdown("</div>", unsafe_allow_html=True)
