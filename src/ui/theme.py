"""Dark-mode theming for the Streamlit app.

Exposes a single public function :func:`render_theme_toggle` that:

* Initialises ``st.session_state['theme']`` to ``'light'`` if unset.
* Injects CSS overrides for ``:root[data-theme='dark']`` so every existing
  component (metric cards, answer card, source cards, sidebar, hero badge,
  tabs, footer, formula box, pipeline flow, latency bar) remains readable
  in dark mode.
* Renders a small 🌙 / ☀️ toggle button in the top-right of the main area.
* Emits a tiny JS snippet via ``st.components.v1.html`` that stamps
  ``data-theme`` on the parent document's ``<html>`` element on every
  Streamlit re-run, so the selected theme persists without a page reload.

Only presentational — no pipeline or retrieval logic is touched.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

SESSION_KEY = "theme"
_VALID = ("light", "dark")


def _current_theme() -> str:
    theme = st.session_state.get(SESSION_KEY, "light")
    return theme if theme in _VALID else "light"


def _toggle_theme() -> None:
    """Flip the theme in session state (button on_click callback)."""
    st.session_state[SESSION_KEY] = "dark" if _current_theme() == "light" else "light"


_DARK_CSS = """
<style>
    /* Dark-mode variable overrides — wired through the existing tokens. */
    :root[data-theme='dark'] {
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --border: #334155;
    }

    /* Streamlit app shell */
    html[data-theme='dark'], html[data-theme='dark'] body,
    html[data-theme='dark'] .stApp,
    html[data-theme='dark'] [data-testid="stAppViewContainer"],
    html[data-theme='dark'] [data-testid="stHeader"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
    }
    html[data-theme='dark'] [data-testid="stSidebar"],
    html[data-theme='dark'] [data-testid="stSidebar"] > div {
        background-color: #1e293b !important;
        color: #f8fafc !important;
    }
    html[data-theme='dark'] [data-testid="stSidebar"] * { color: #e2e8f0; }

    /* Generic text elements */
    html[data-theme='dark'] p,
    html[data-theme='dark'] span,
    html[data-theme='dark'] label,
    html[data-theme='dark'] li,
    html[data-theme='dark'] h1, html[data-theme='dark'] h2,
    html[data-theme='dark'] h3, html[data-theme='dark'] h4,
    html[data-theme='dark'] h5, html[data-theme='dark'] h6 {
        color: #f8fafc;
    }

    /* Hero */
    html[data-theme='dark'] .hero-subtitle { color: #cbd5e1; }

    /* Pipeline flow container */
    html[data-theme='dark'] .pipeline-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-color: #334155;
    }
    html[data-theme='dark'] .flow-step {
        background: #0f172a;
        border-color: #334155;
        color: #e2e8f0;
    }
    html[data-theme='dark'] .flow-step:hover {
        border-color: #818cf8;
        color: #a5b4fc;
    }
    html[data-theme='dark'] .flow-arrow { color: #475569; }

    /* Section headers */
    html[data-theme='dark'] .section-title { color: #f8fafc; }

    /* Answer card — keep green identity but on dark surface */
    html[data-theme='dark'] .answer-container {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border-color: #10b981;
    }
    html[data-theme='dark'] .answer-header-text { color: #6ee7b7; }
    html[data-theme='dark'] .answer-text { color: #f0fdf4; }

    /* Source cards */
    html[data-theme='dark'] .source-card {
        background: #1e293b;
        border-color: #334155;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4);
    }
    html[data-theme='dark'] .source-card:hover {
        border-color: #818cf8;
        box-shadow: 0 4px 12px rgba(129, 140, 248, 0.2);
    }
    html[data-theme='dark'] .source-content { color: #e2e8f0; }
    html[data-theme='dark'] .source-meta {
        color: #94a3b8;
        border-top-color: #334155;
    }
    html[data-theme='dark'] .source-rank { color: #a5b4fc; }

    /* Formula / latency boxes */
    html[data-theme='dark'] .formula-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-color: #334155;
        color: #e2e8f0;
    }
    html[data-theme='dark'] .latency-bar {
        background: linear-gradient(135deg, #312e81 0%, #4c1d95 100%);
        border-color: #6d28d9;
    }
    html[data-theme='dark'] .latency-value { color: #a5b4fc; }
    html[data-theme='dark'] .latency-label { color: #c4b5fd; }

    /* Sidebar config cards */
    html[data-theme='dark'] .sidebar-section {
        background: #0f172a;
        border-color: #334155;
        border-left-color: #818cf8;
    }
    html[data-theme='dark'] .sidebar-label { color: #a5b4fc; }
    html[data-theme='dark'] .sidebar-value { color: #f8fafc; }

    /* Query container */
    html[data-theme='dark'] .query-container {
        background: #1e293b;
        border-color: #334155;
    }
    html[data-theme='dark'] .query-container:focus-within { border-color: #818cf8; }

    /* Tabs */
    html[data-theme='dark'] .stTabs [data-baseweb="tab-list"] {
        background: #1e293b;
    }
    html[data-theme='dark'] .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {
        color: #cbd5e1 !important;
    }
    html[data-theme='dark'] .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: #0f172a !important;
        color: #f8fafc !important;
    }

    /* Inputs */
    html[data-theme='dark'] input,
    html[data-theme='dark'] textarea,
    html[data-theme='dark'] [data-baseweb="input"] input,
    html[data-theme='dark'] [data-baseweb="textarea"] textarea {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-color: #334155 !important;
    }

    /* Buttons — keep primary colour, darken secondary */
    html[data-theme='dark'] .stButton > button {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
    }
    html[data-theme='dark'] .stButton > button:hover {
        border-color: #818cf8;
        color: #a5b4fc;
    }
    html[data-theme='dark'] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
    }

    /* Footer */
    html[data-theme='dark'] .footer { border-top-color: #334155; }
    html[data-theme='dark'] .footer-text { color: #94a3b8; }
    html[data-theme='dark'] .footer-link { color: #a5b4fc; }

    /* Code blocks */
    html[data-theme='dark'] pre, html[data-theme='dark'] code {
        background-color: #0f172a !important;
        color: #e2e8f0 !important;
    }

    /* Toggle button — floats top-right of the main area */
    .theme-toggle-slot { position: relative; }
    div[data-testid="stHorizontalBlock"] .stButton > button.theme-toggle,
    .stButton > button[data-theme-toggle="1"] {
        border-radius: 999px !important;
        padding: 0.35rem 0.7rem !important;
    }
</style>
"""


def _inject_data_theme_js(theme: str) -> None:
    """Set ``data-theme`` on the parent document's <html> element.

    Uses ``st.components.v1.html`` so we can reach out of the Streamlit
    iframe via ``window.parent.document``. The component renders zero
    visible height.
    """
    safe = "dark" if theme == "dark" else "light"
    components.html(
        f"""
        <script>
        (function() {{
            try {{
                var doc = window.parent && window.parent.document
                    ? window.parent.document
                    : document;
                doc.documentElement.setAttribute('data-theme', '{safe}');
            }} catch (e) {{
                document.documentElement.setAttribute('data-theme', '{safe}');
            }}
        }})();
        </script>
        """,
        height=0,
    )


def render_theme_toggle() -> str:
    """Render the theme toggle and apply the current theme.

    Call this once, near the top of the app (after ``st.set_page_config``
    and after the base CSS block). Returns the active theme name.
    """
    if SESSION_KEY not in st.session_state or st.session_state[SESSION_KEY] not in _VALID:
        st.session_state[SESSION_KEY] = "light"

    # Inject dark-mode overrides once per run — cheap and idempotent.
    st.markdown(_DARK_CSS, unsafe_allow_html=True)

    theme = _current_theme()

    # Right-aligned toggle button. spacer column pushes it to the edge.
    spacer, btn_col = st.columns([12, 1])
    with btn_col:
        label = "☀️" if theme == "dark" else "🌙"
        st.button(
            label,
            key="__theme_toggle_btn",
            help="Toggle dark mode",
            on_click=_toggle_theme,
            use_container_width=True,
        )

    # After the (possible) on_click flip, read the up-to-date value.
    theme = _current_theme()
    _inject_data_theme_js(theme)
    return theme
