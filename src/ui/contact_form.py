"""Contact form UI section for the Streamlit app.

Renders a styled card matching the existing gradient / --primary /
--bg-card / --radius-lg aesthetic, collects Name / Email / Message,
and delegates to :func:`src.contact.handler.handle_contact_submission`.

Behaviour:

* Empty fields  -> inline ``st.warning`` (handler not called).
* Bad email     -> inline ``st.error`` (handler not called).
* Success       -> green banner replaces the form and clears fields.
* Backend fail  -> red banner, values retained for retry.

Only presentational glue — no pipeline / retrieval logic touched.
"""

from __future__ import annotations

import streamlit as st

from src.contact.handler import (
    ContactSubmissionError,
    handle_contact_submission,
    is_valid_email,
)

_SENT_KEY = "__contact_form_sent"
_NAME_KEY = "__contact_form_name"
_EMAIL_KEY = "__contact_form_email"
_MESSAGE_KEY = "__contact_form_message"


_CONTACT_CSS = """
<style>
    .contact-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-left: 4px solid var(--primary, #6366f1);
        border-radius: var(--radius-lg, 16px);
        padding: 1.5rem 2rem;
        margin: 1rem 0 2rem;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08);
    }
    .contact-card .contact-lede {
        color: #475569;
        font-size: 0.95rem;
        margin: -0.25rem 0 0.75rem;
    }
    .contact-success {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1px solid #bbf7d0;
        border-left: 4px solid #10b981;
        border-radius: var(--radius-lg, 16px);
        padding: 1.25rem 1.75rem;
        margin: 1rem 0 2rem;
        color: #166534;
        font-weight: 600;
    }
    html[data-theme='dark'] .contact-card {
        background: linear-gradient(135deg, var(--bg-card, #1e293b) 0%, #0f172a 100%);
        border-color: #334155;
        border-left-color: #818cf8;
    }
    html[data-theme='dark'] .contact-card .contact-lede { color: #cbd5e1; }
    html[data-theme='dark'] .contact-success {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border-color: #10b981;
        color: #d1fae5;
    }
</style>
"""


def _reset_form_state() -> None:
    for key in (_NAME_KEY, _EMAIL_KEY, _MESSAGE_KEY):
        st.session_state.pop(key, None)


def render_contact_form() -> None:
    """Render the contact form section.

    Safe to call unconditionally near the bottom of ``app.py``, above the
    footer. All UI state is namespaced under ``__contact_form_*`` keys
    so it cannot collide with the rest of the app.
    """
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

    # Success state: banner replaces the form until the user dismisses it.
    if st.session_state.get(_SENT_KEY):
        st.markdown(
            '<div class="contact-success">✅ Message sent! '
            "Thanks for reaching out — I'll get back to you soon.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Send another message", key="__contact_reset_btn"):
            st.session_state[_SENT_KEY] = False
            _reset_form_state()
            st.rerun()
        return

    st.markdown('<div class="contact-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="contact-lede">Have a question, a bug report, or want to '
        "collaborate? Drop a message below.</div>",
        unsafe_allow_html=True,
    )

    with st.form("contact_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            name = st.text_input("Name", key=_NAME_KEY, placeholder="Jane Doe")
        with col_b:
            email = st.text_input(
                "Email", key=_EMAIL_KEY, placeholder="you@example.com"
            )
        message = st.text_area(
            "Message",
            key=_MESSAGE_KEY,
            placeholder="How can I help?",
            height=140,
        )
        submitted = st.form_submit_button("Submit", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        return

    # ── Client-side validation (mirrors handler._validate) ──────────────
    name_v = (name or "").strip()
    email_v = (email or "").strip()
    message_v = (message or "").strip()

    if not name_v:
        st.warning("Name is required.")
        return
    if not email_v:
        st.warning("Email is required.")
        return
    if not is_valid_email(email_v):
        st.error("Email address looks invalid.")
        return
    if not message_v:
        st.warning("Message is required.")
        return

    # ── Delegate to the handler ─────────────────────────────────────────
    try:
        handle_contact_submission(name_v, email_v, message_v)
    except ValueError as exc:
        # Defensive: handler validation is stricter than ours may become.
        st.error(str(exc))
        return
    except ContactSubmissionError as exc:
        st.error(f"Sorry — we couldn't deliver your message: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 — never crash the app
        st.error(f"Unexpected error: {exc}")
        return

    # Success — clear fields and flip to the banner on next run.
    _reset_form_state()
    st.session_state[_SENT_KEY] = True
    st.rerun()
