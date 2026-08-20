"""Contact form submission handler (stub).

Story 1 (this story) only wires up the UI. Story 2 will replace the body
of :func:`handle_contact_submission` with the real delivery mechanism
(SMTP / SendGrid / webhook). The signature and exception contract
documented here are the stable interface the UI relies on.

Contract
--------
* All three fields are required, non-empty strings.
* ``email`` must contain at least one ``@`` character (basic format check).
* Raises ``ValueError`` for client-side validation failures.
* Raises ``ContactSubmissionError`` for delivery / backend failures.
* Returns ``None`` on success.
"""

from __future__ import annotations

import re
from typing import Final

# Pragmatic, deliberately-lenient email pattern: ``something@something``.
# The acceptance criteria only require an ``@`` check; we additionally
# require at least one character on each side and no whitespace so we
# don't accept obviously-broken values like ``"@"`` or ``"a b@c"``.
_EMAIL_RE: Final = re.compile(r"^[^\s@]+@[^\s@]+$")


class ContactSubmissionError(RuntimeError):
    """Raised when a validated submission cannot be delivered."""


def is_valid_email(email: str) -> bool:
    """Return True if *email* passes the lightweight client-side check."""
    if not isinstance(email, str):
        return False
    return bool(_EMAIL_RE.match(email.strip()))


def _validate(name: str, email: str, message: str) -> tuple[str, str, str]:
    name = (name or "").strip()
    email = (email or "").strip()
    message = (message or "").strip()

    if not name:
        raise ValueError("Name is required.")
    if not email:
        raise ValueError("Email is required.")
    if not is_valid_email(email):
        raise ValueError("Email address looks invalid.")
    if not message:
        raise ValueError("Message is required.")

    return name, email, message


def handle_contact_submission(name: str, email: str, message: str) -> None:
    """Receive a contact form submission.

    Story 1 stub: validates inputs and logs the submission. Story 2 will
    replace the body with actual delivery (e.g. SMTP / SendGrid). The
    function signature, validation rules and exception contract MUST be
    preserved so the UI continues to work unchanged.
    """
    _validate(name, email, message)
    # Story 2: replace this no-op with real delivery and raise
    # ``ContactSubmissionError`` on backend failure.
    return None
