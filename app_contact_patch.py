"""
Integration shim for the contact form.

Usage — in ``app.py``, immediately before the ``# ─── Footer ──`` block, add::

    from app_contact_patch import mount_contact_form
    mount_contact_form()

Kept as a tiny separate module so ``app.py`` only needs a two-line edit.
"""

from src.contact import render_contact_form


def mount_contact_form() -> None:
    """Render the contact form on the current Streamlit page."""
    render_contact_form()
