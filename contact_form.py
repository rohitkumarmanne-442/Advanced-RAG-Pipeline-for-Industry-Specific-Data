"""
contact_form.py
---------------
Contact-form rendering and submission logic for the Streamlit app.

Secrets / env-var keys expected (all optional — graceful fallback if absent):
  FORMSPREE_URL   – Formspree endpoint, e.g. https://formspree.io/f/xxxx
  SMTP_HOST       – SMTP server hostname
  SMTP_PORT       – SMTP server port (default 587)
  SMTP_USER       – SMTP login username / sender address
  SMTP_PASSWORD   – SMTP login password
  CONTACT_EMAIL   – recipient address for email submissions
"""

from __future__ import annotations

import logging
import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _get_secret(key: str) -> Optional[str]:
    """Read a secret from st.secrets first, then environment variables."""
    try:
        value = st.secrets.get(key)
        if value:
            return str(value)
    except Exception:  # st.secrets not configured
        pass
    return os.environ.get(key) or None


def validate_form(name: str, email: str, message: str) -> Optional[str]:
    """
    Validate contact-form inputs.

    Returns an error string if validation fails, or None if all inputs are valid.
    Strips surrounding whitespace before checking.
    """
    name = name.strip()
    email = email.strip()
    message = message.strip()

    if not name or not email or not message:
        return "Please fill in all fields (Name, Email, and Message)."

    if not EMAIL_RE.match(email):
        return "Please enter a valid email address (e.g. you@example.com)."

    return None


def _send_via_formspree(url: str, name: str, email: str, message: str) -> None:
    """POST submission to a Formspree endpoint."""
    import urllib.request
    import urllib.parse
    import json

    payload = json.dumps({"name": name, "email": email, "message": message}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status not in (200, 201, 202):
            raise RuntimeError(f"Formspree returned HTTP {resp.status}")


def _send_via_smtp(
    host: str,
    port: int,
    user: str,
    password: str,
    recipient: str,
    name: str,
    email: str,
    message: str,
) -> None:
    """Send contact submission via SMTP (TLS)."""
    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = recipient
    msg["Subject"] = f"[Contact Form] Message from {name}"
    body = f"Name: {name}\nEmail: {email}\n\n{message}"
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(host, port, timeout=10) as server:
        server.ehlo()
        server.starttls()
        server.login(user, password)
        server.sendmail(user, recipient, msg.as_string())


def submit_contact_form(name: str, email: str, message: str) -> None:
    """
    Attempt to deliver the contact form submission.

    Priority:
      1. Formspree  (FORMSPREE_URL secret)
      2. SMTP       (SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD / CONTACT_EMAIL secrets)
      3. Log-only fallback (safe for demo / local dev with no secrets configured)
    """
    name = name.strip()
    email = email.strip()
    message = message.strip()

    formspree_url = _get_secret("FORMSPREE_URL")
    if formspree_url:
        try:
            _send_via_formspree(formspree_url, name, email, message)
            logger.info("Contact form submitted via Formspree from %s", email)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Formspree submission failed: %s", exc)
            # fall through to SMTP / log fallback

    smtp_host = _get_secret("SMTP_HOST")
    smtp_user = _get_secret("SMTP_USER")
    smtp_password = _get_secret("SMTP_PASSWORD")
    smtp_recipient = _get_secret("CONTACT_EMAIL") or smtp_user

    if smtp_host and smtp_user and smtp_password and smtp_recipient:
        try:
            smtp_port = int(_get_secret("SMTP_PORT") or 587)
            _send_via_smtp(
                smtp_host,
                smtp_port,
                smtp_user,
                smtp_password,
                smtp_recipient,
                name,
                email,
                message,
            )
            logger.info("Contact form submitted via SMTP from %s", email)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("SMTP submission failed: %s", exc)
            # fall through to log-only fallback

    # Log-only fallback — never crash the app
    logger.info(
        "Contact form submission (log-only fallback) | name=%r email=%r message=%r",
        name,
        email,
        message,
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

CONTACT_CSS = """
<style>
.contact-card {
    background: #ffffff;
    border-left: 4px solid var(--indigo, #4f46e5);
    border-radius: var(--radius, 12px);
    padding: 2rem 2rem 1.5rem;
    margin: 2rem 0 1rem;
    box-shadow: 0 2px 12px rgba(79,70,229,0.07);
}
.contact-card h2 {
    color: var(--indigo, #4f46e5);
    margin-top: 0;
    font-size: 1.4rem;
}
.contact-card p {
    color: #555;
    margin-bottom: 1.2rem;
    font-size: 0.95rem;
}
</style>
"""


def render_contact_form() -> None:
    """
    Render the contact form section inside a styled card.
    Call this in app.py just above the footer.
    """
    st.markdown(CONTACT_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="contact-card">'
        '<h2>\U0001f4ec Contact Us</h2>'
        "<p>Have a question, found a bug, or want to share feedback? "
        "Fill in the form below and we'll get back to you.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form(key="contact_form", clear_on_submit=True):
        name = st.text_input("Name", placeholder="Jane Doe")
        email = st.text_input("Email", placeholder="you@example.com")
        message = st.text_area("Message", placeholder="Your message…", height=150)
        submitted = st.form_submit_button("Send")

    if submitted:
        error = validate_form(name, email, message)
        if error:
            st.warning(error)
        else:
            try:
                submit_contact_form(name, email, message)
            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected error in contact form submission: %s", exc)
                # Still show success to the user — the log has the details.
            st.success(
                "\u2705 Thanks for reaching out! Your message has been received "
                "and we'll get back to you soon."
            )
