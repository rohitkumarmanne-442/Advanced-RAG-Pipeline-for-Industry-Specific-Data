"""One-line integration shim for app.py.

Usage in app.py (add immediately BEFORE the existing footer markdown block):

    from src.contact.app_integration import mount_contact_section
    mount_contact_section()

Kept as a separate module so app.py only needs a single import + call —
any future contact-form changes stay isolated inside src/contact/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.contact.contact_form import render_contact_form


def mount_contact_section(log_path: Optional[Path] = None) -> None:
    """Render the contact form section. Safe to call unconditionally."""
    render_contact_form(log_path=log_path)
