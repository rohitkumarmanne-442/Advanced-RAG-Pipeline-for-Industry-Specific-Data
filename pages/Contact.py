"""
Contact Page - Streamlit Multipage

Renders a styled contact form that matches the visual design language of the
main RAG Pipeline app (gradient headers, card containers, shared CSS variables).

This page is auto-discovered by Streamlit via the `pages/` directory convention
and appears in the sidebar navigation. No backend submission is wired up yet —
that is covered by a follow-up story; the submit button currently shows a
friendly confirmation toast and basic client-side validation feedback.
"""

import os
import re

import streamlit as st

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Contact | Advanced RAG Pipeline",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Suppress telemetry consistently with app.py
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


# ─── Constants ────────────────────────────────────────────────────────────────

SUBJECT_OPTIONS = [
    "General Inquiry",
    "Bug Report",
    "Feature Request",
    "Partnership",
    "Other",
]

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ─── Custom CSS (mirrors app.py design tokens) ────────────────────────────────

st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --secondary: #ec4899;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border: #334155;
    }

    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #ec4899, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 0.5rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 0.35rem 0.9rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    /* Contact card container — matches pipeline-container styling */
    .contact-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem 2.25rem;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .contact-card-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1rem;
    }
    .contact-card-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: #ede9fe;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .contact-card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
    }
    .contact-card-help {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 0.5rem;
    }

    .stButton > button {
        border-radius: 10px !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.2rem !important;
        font-weight: 600 !important;
    }

    .footer {
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    .footer-text {
        color: #94a3b8;
        font-size: 0.82rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Hero ─────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero-container">
    <div class="hero-title">Get in Touch</div>
    <div class="hero-subtitle">Questions, feedback, or ideas? We'd love to hear from you.</div>
    <div class="hero-badge">✉️ Contact the Team</div>
</div>
""",
    unsafe_allow_html=True,
)


# ─── Contact Card + Form ──────────────────────────────────────────────────────

st.markdown(
    """
<div class="contact-card">
    <div class="contact-card-header">
        <div class="contact-card-icon">💬</div>
        <div class="contact-card-title">Send us a message</div>
    </div>
    <div class="contact-card-help">
        Fill out the form below and we'll get back to you as soon as possible.
        All fields are required.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.form("contact_form", clear_on_submit=False):
    col_name, col_email = st.columns(2)
    with col_name:
        name = st.text_input(
            "Name",
            max_chars=120,
            placeholder="Jane Doe",
            help="Your full name.",
        )
    with col_email:
        email = st.text_input(
            "Email",
            max_chars=254,
            placeholder="jane.doe@example.com",
            help="We'll only use this to reply to your message.",
        )

    subject = st.selectbox(
        "Subject",
        options=SUBJECT_OPTIONS,
        index=0,
        help="Pick the topic that best matches your message.",
    )

    message = st.text_area(
        "Message",
        max_chars=2000,
        height=180,
        placeholder="Tell us what's on your mind...",
        help="Up to 2,000 characters.",
    )

    submitted = st.form_submit_button("Send Message", type="primary")

if submitted:
    errors = []
    name_clean = (name or "").strip()
    email_clean = (email or "").strip()
    message_clean = (message or "").strip()

    if not name_clean:
        errors.append("Name is required.")
    if not email_clean:
        errors.append("Email is required.")
    elif not EMAIL_REGEX.match(email_clean):
        errors.append("Please enter a valid email address.")
    if subject not in SUBJECT_OPTIONS:
        errors.append("Please choose a valid subject.")
    if not message_clean:
        errors.append("Message cannot be empty.")
    elif len(message_clean) < 10:
        errors.append("Message is a bit short — please add a little more detail.")

    if errors:
        for err in errors:
            st.error(err)
    else:
        # Backend submission is intentionally deferred to a follow-up story.
        st.success(
            f"Thanks, {name_clean}! Your message has been queued. "
            "We'll reply to " + email_clean + " shortly."
        )


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="footer">
    <div class="footer-text">
        <strong>Advanced RAG Pipeline</strong> &mdash; We typically respond within 1–2 business days.
    </div>
</div>
""",
    unsafe_allow_html=True,
)
