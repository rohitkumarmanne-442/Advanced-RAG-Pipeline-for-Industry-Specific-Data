"""Regression tests enforcing removal of the contact form.

These tests fail loudly if the deleted modules are reintroduced or if
``app.py`` grows a new reference to them.
"""

import importlib
from pathlib import Path

import pytest


def test_contact_form_ui_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.ui.contact_form")


def test_contact_handler_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.contact.handler")


def test_contact_package_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.contact")


def test_app_py_has_no_contact_form_references():
    app_py = Path(__file__).resolve().parent.parent / "app.py"
    source = app_py.read_text(encoding="utf-8")
    assert "contact_form" not in source, "app.py still references contact_form"
    assert "render_contact_form" not in source, (
        "app.py still calls render_contact_form"
    )
    assert "Get in Touch" not in source, "app.py still renders 'Get in Touch'"
