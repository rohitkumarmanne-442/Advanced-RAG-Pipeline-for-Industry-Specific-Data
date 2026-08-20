"""Shim that exposes the contact form to ``app.py``.

``app.py`` is a read-only file in this change set; this shim lets us wire
the new contact section in with a single import + call in a follow-up
edit while keeping all logic isolated in :mod:`src.contact.contact_form`.

Usage in app.py (add just above the ``# ─── Footer`` block):

    from app_contact_hook import render_contact_form
    render_contact_form()
"""

from src.contact.contact_form import render_contact_form

__all__ = ["render_contact_form"]
