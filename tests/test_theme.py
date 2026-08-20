"""Tests for the dark-mode toggle helper (src/ui/theme.py).

We exercise the pure state logic directly. Streamlit's rendering path is
out of scope for unit tests — those are covered manually per the PR test
plan.
"""

import sys
import types

import pytest


@pytest.fixture
def fake_streamlit(monkeypatch):
    """Install a minimal fake ``streamlit`` + ``streamlit.components.v1``.

    Enough surface area for :mod:`src.ui.theme` to import and for us to
    drive ``_toggle_theme`` / ``_current_theme`` without a real Streamlit
    runtime.
    """
    st_mod = types.ModuleType("streamlit")
    st_mod.session_state = {}

    def _noop(*_a, **_kw):
        return None

    def _columns(spec):
        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return tuple(_Ctx() for _ in range(n))

    st_mod.markdown = _noop
    st_mod.button = _noop
    st_mod.columns = _columns

    comp_pkg = types.ModuleType("streamlit.components")
    comp_v1 = types.ModuleType("streamlit.components.v1")
    comp_v1.html = _noop
    comp_pkg.v1 = comp_v1

    monkeypatch.setitem(sys.modules, "streamlit", st_mod)
    monkeypatch.setitem(sys.modules, "streamlit.components", comp_pkg)
    monkeypatch.setitem(sys.modules, "streamlit.components.v1", comp_v1)

    # Ensure a fresh import of the module under test picks up the fakes.
    sys.modules.pop("src.ui.theme", None)
    import src.ui.theme as theme_mod  # noqa: WPS433 — re-import intentional

    return st_mod, theme_mod


def test_default_theme_is_light(fake_streamlit):
    st_mod, theme_mod = fake_streamlit
    st_mod.session_state.clear()
    assert theme_mod._current_theme() == "light"


def test_toggle_flips_light_to_dark(fake_streamlit):
    st_mod, theme_mod = fake_streamlit
    st_mod.session_state.clear()
    st_mod.session_state["theme"] = "light"

    theme_mod._toggle_theme()

    assert st_mod.session_state["theme"] == "dark"


def test_toggle_flips_dark_back_to_light(fake_streamlit):
    st_mod, theme_mod = fake_streamlit
    st_mod.session_state.clear()
    st_mod.session_state["theme"] = "dark"

    theme_mod._toggle_theme()

    assert st_mod.session_state["theme"] == "light"


def test_invalid_session_value_is_treated_as_light(fake_streamlit):
    st_mod, theme_mod = fake_streamlit
    st_mod.session_state.clear()
    st_mod.session_state["theme"] = "neon"
    assert theme_mod._current_theme() == "light"


def test_render_writes_default_into_session_state(fake_streamlit):
    st_mod, theme_mod = fake_streamlit
    st_mod.session_state.clear()

    result = theme_mod.render_theme_toggle()

    assert result == "light"
    assert st_mod.session_state["theme"] == "light"


def test_render_preserves_dark_across_reruns(fake_streamlit):
    """Simulates AC #4: theme survives a Streamlit re-run."""
    st_mod, theme_mod = fake_streamlit
    st_mod.session_state.clear()
    st_mod.session_state["theme"] = "dark"

    # Re-run: render is called again; session state must be honoured.
    result = theme_mod.render_theme_toggle()

    assert result == "dark"
    assert st_mod.session_state["theme"] == "dark"
