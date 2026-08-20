"""Contact form submission handling."""

from src.contact.handler import (
    ContactSubmissionError,
    handle_contact_submission,
    is_valid_email,
)

__all__ = [
    "ContactSubmissionError",
    "handle_contact_submission",
    "is_valid_email",
]
