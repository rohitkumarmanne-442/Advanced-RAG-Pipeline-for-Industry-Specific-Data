"""Contact form page for the Advanced RAG Pipeline Streamlit app.

Renders a styled contact form (Name, Email, Subject, Message) with
client-side validation:
  * All fields are required.
  * Email must be a well-formed address (RFC 5322-ish regex).
  * Message must be at least 20 characters.

On a valid submission the form is cleared and a green success banner is
shown. On validation failure, inline ``st.error`` hints are rendered
next to each offending field.

The page is designed to slot into ``app.py``'s existing indigo/slate
design tokens (primary ``#6366f1``, slate text ``#1e293b`` / ``#94a3b8``).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBJECT_CATEGORIES: List[str] = [
    "",  # placeholder so an unselected dropdown is treated as empty
    "General Question",
    "Bug Report",
    "Feature Request",
    "Partnership Enquiry",
    "Feedback",
]

MIN_MESSAGE_LENGTH: int = 20
MAX_MESSAGE_LENGTH: int = 2000

# A pragmatic email regex: requires a local part, an '@', a domain label
# and a TLD of at least 2 characters. Good enough for client-side checks;
# real delivery validation happens server-side.
_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)

# Session-state keys (namespaced to avoid clashes with the rest of app.py)
_K_NAME = "contact_form_name"
_K_EMAIL = "contact_form_email"
_K_SUBJECT = "contact_form_subject"
_K_MESSAGE = "contact_form_message"
_K_SUBMITTED_OK = "contact_form_submitted_ok"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_valid_email(email: str) -> bool:
    """Return True if ``email`` looks like a valid address."""
    if not email:
        return False
    return bool(_EMAIL_RE.match(email.strip()))


def validate_contact_form(
    name: str,
    email: str,
    subject: str,
    message: str,
) -> Dict[str, str]:
    """Validate the contact form input.

    Returns a dict mapping field name -> human-readable error message.
    An empty dict means the submission is valid.
    """
    errors: Dict[str, str] = {}

    if not name or not name.strip():
        errors["name"] = "Name is required"

    if not email or not email.strip():
        errors["email"] = "Email is required"
    elif not _is_valid_email(email):
        errors["email"] = "Please enter a valid email address"

    if not subject or not subject.strip():
        errors["subject"] = "Please choose a subject category"

    if not message or not message.strip():
        errors["message"] = "Message is required"
    elif len(message.strip()) < MIN_MESSAGE_LENGTH:
        errors["message"] = (
            f"Message must be at least {MIN_MESSAGE_LENGTH} characters"
        )

    return errors


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_CONTACT_CSS = """
<style>
    .contact-hero {
        text-align: center;
        padding: 1.5rem 1rem 0.5rem;
    }
    .contact-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .contact-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .contact-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.5rem;
    }
    .contact-counter {
        text-align: right;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: -0.4rem;
        margin-bottom: 0.6rem;
    }
    .contact-counter.warn { color: #f59e0b; }
    .contact-counter.ok   { color: #10b981; }
</style>
"""


def _ensure_state() -> None:
    """Initialise session-state keys if they don't exist yet."""
    st.session_state.setdefault(_K_NAME, "")
    st.session_state.setdefault(_K_EMAIL, "")
    st.session_state.setdefault(_K_SUBJECT, SUBJECT_CATEGORIES[0])
    st.session_state.setdefault(_K_MESSAGE, "")
    st.session_state.setdefault(_K_SUBMITTED_OK, False)


def _clear_form() -> None:
    """Reset all contact-form fields in session state."""
    st.session_state[_K_NAME] = ""
    st.session_state[_K_EMAIL] = ""
    st.session_state[_K_SUBJECT] = SUBJECT_CATEGORIES[0]
    st.session_state[_K_MESSAGE] = ""


def _counter_class(length: int) -> str:
    if length == 0:
        return ""
    if length < MIN_MESSAGE_LENGTH:
        return "warn"
    return "ok"


def render_contact_page() -> None:
    """Render the full Contact Us page.

    Call this from ``app.py`` when the user selects 'Contact Us' from
    the sidebar navigation. Safe to call repeatedly across reruns.
    """
    _ensure_state()
    st.markdown(_CONTACT_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="contact-hero">
            <div class="contact-title">Contact Us</div>
            <div class="contact-subtitle">
                Questions, bug reports or partnership enquiries — drop us a line.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show post-submit success banner from previous run, then clear the flag
    if st.session_state.get(_K_SUBMITTED_OK):
        st.success("✅ Your message has been sent!")
        st.session_state[_K_SUBMITTED_OK] = False

    with st.container():
        st.markdown('<div class="contact-card">', unsafe_allow_html=True)

        # We deliberately do NOT use st.form because we want live char
        # counter updates on the message textarea.
        col_name, col_email = st.columns(2)
        with col_name:
            name = st.text_input(
                "Name *",
                key=_K_NAME,
                placeholder="Jane Doe",
                max_chars=120,
            )
            name_err_slot = st.empty()
        with col_email:
            email = st.text_input(
                "Email *",
                key=_K_EMAIL,
                placeholder="jane@example.com",
                max_chars=254,
            )
            email_err_slot = st.empty()

        subject = st.selectbox(
            "Subject *",
            options=SUBJECT_CATEGORIES,
            key=_K_SUBJECT,
            format_func=lambda v: "— Choose a category —" if v == "" else v,
        )
        subject_err_slot = st.empty()

        message = st.text_area(
            "Message *",
            key=_K_MESSAGE,
            placeholder=(
                "Tell us what's on your mind — minimum "
                f"{MIN_MESSAGE_LENGTH} characters."
            ),
            height=180,
            max_chars=MAX_MESSAGE_LENGTH,
        )
        msg_len = len(message or "")
        st.markdown(
            f'<div class="contact-counter {_counter_class(msg_len)}">'
            f"{msg_len} / {MAX_MESSAGE_LENGTH} characters "
            f"(min {MIN_MESSAGE_LENGTH})"
            "</div>",
            unsafe_allow_html=True,
        )
        message_err_slot = st.empty()

        send_clicked = st.button(
            "✉️ Send Message",
            type="primary",
            use_container_width=False,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    if not send_clicked:
        return

    errors = validate_contact_form(name, email, subject, message)

    if errors:
        if "name" in errors:
            name_err_slot.error(errors["name"])
        if "email" in errors:
            email_err_slot.error(errors["email"])
        if "subject" in errors:
            subject_err_slot.error(errors["subject"])
        if "message" in errors:
            message_err_slot.error(errors["message"])
        return

    # Valid submission — in a real deployment this is where we'd POST to
    # a backend / send an email. For now we just record the success and
    # clear the form, then rerun so the cleared widgets render.
    _persist_submission(name=name, email=email, subject=subject, message=message)
    _clear_form()
    st.session_state[_K_SUBMITTED_OK] = True
    st.rerun()


def _persist_submission(*, name: str, email: str, subject: str, message: str) -> None:
    """Hook for persisting a submission.

    Kept as a separate function so it can be swapped for an email / DB
    integration without touching the UI code. Currently a no-op that
    only stores the last submission in session state for debugging.
    """
    st.session_state["contact_form_last_submission"] = {
        "name": name.strip(),
        "email": email.strip(),
        "subject": subject,
        "message": message.strip(),
    }
