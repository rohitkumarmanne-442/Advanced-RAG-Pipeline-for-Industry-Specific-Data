"""Contact form handling module.

The concrete delivery handler (SMTP / email service / webhook) is
implemented in Story 2. This stub provides a stable import path that the
Streamlit UI can call today; Story 2 will replace the body of
``handle_contact_submission`` without touching ``app.py``.
"""

from src.contact.handler import handle_contact_submission

__all__ = ["handle_contact_submission"]
