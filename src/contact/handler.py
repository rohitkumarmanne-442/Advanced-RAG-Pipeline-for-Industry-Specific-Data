"""Contact form submission handler.

Validates a name/email/message payload and appends valid submissions to a
local ``contacts.log`` file as JSON lines. Kept intentionally dependency-free
so it can be imported from the Streamlit app without pulling in the RAG stack.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# RFC 5322 is overkill for a contact form; this is the pragmatic pattern:
# one or more non-space/@ chars, an @, one or more non-space/@ chars,
# optionally followed by .tld-ish segments. We tolerate short TLD-less
# addresses like ``a@b`` for internal testing, matching the unit tests.
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+$")

# Default log location — project root / contacts.log. Overridable via env var
# for tests and alternate deployments.
_DEFAULT_LOG_PATH = Path(__file__).resolve().parents[2] / "contacts.log"

# Reasonable upper bounds so a malicious submitter cannot balloon the log.
_MAX_NAME_LEN = 200
_MAX_EMAIL_LEN = 320  # per RFC 5321
_MAX_MESSAGE_LEN = 5000


class ContactSubmissionError(RuntimeError):
    """Raised when a validated submission cannot be persisted/delivered."""


def is_valid_email(email) -> bool:
    """Return True if ``email`` looks like a valid address.

    Non-string inputs, empty strings, and obvious malformations return False.
    Surrounding whitespace is tolerated (the caller usually shows a trimmed
    version back to the user).
    """
    if not isinstance(email, str):
        return False
    candidate = email.strip()
    if not candidate or len(candidate) > _MAX_EMAIL_LEN:
        return False
    return bool(_EMAIL_RE.match(candidate))


def _log_path() -> Path:
    override = os.environ.get("CONTACT_LOG_PATH")
    return Path(override) if override else _DEFAULT_LOG_PATH


def handle_contact_submission(
    name: str,
    email: str,
    message: str,
    *,
    log_path: Optional[Path] = None,
) -> None:
    """Validate and persist a contact form submission.

    Raises ``ValueError`` for invalid input (empty fields or bad email) and
    ``ContactSubmissionError`` if the payload cannot be written to disk.
    Returns ``None`` on success.
    """
    # Coerce & trim — treat ``None`` and non-strings as empty.
    name_s = (name or "").strip() if isinstance(name, str) else ""
    email_s = (email or "").strip() if isinstance(email, str) else ""
    message_s = (message or "").strip() if isinstance(message, str) else ""

    if not name_s or not email_s or not message_s:
        raise ValueError("Please fill in all fields.")
    if not is_valid_email(email_s):
        raise ValueError("Please enter a valid email address.")

    # Enforce sane length limits.
    name_s = name_s[:_MAX_NAME_LEN]
    email_s = email_s[:_MAX_EMAIL_LEN]
    message_s = message_s[:_MAX_MESSAGE_LEN]

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name_s,
        "email": email_s,
        "message": message_s,
    }

    target = Path(log_path) if log_path is not None else _log_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:  # pragma: no cover - filesystem-dependent
        raise ContactSubmissionError(
            f"Could not record submission: {exc}"
        ) from exc

    return None
