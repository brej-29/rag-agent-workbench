"""App-wide state shared by every page: backend address, key, and wake-up status."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import httpx
import streamlit as st

from services.backend_client import get_facts, probe_health

# The free host can take a while to boot a sleeping Space; give up after this.
WAKE_LIMIT_S = 180
# Prose stays readable (about 75 characters a line) inside the wide layout that tables and columns need.
PROSE_WIDTH = 720
THEME_CSS = Path(__file__).resolve().parents[1] / "static" / "theme.css"


def inject_theme() -> None:
    """Load the shared, selector-stable visual layer once at the app entry point."""
    dark = DARK_TOKENS if is_dark() else ""
    st.html(f"<style>{THEME_CSS.read_text(encoding='utf-8')}{dark}</style>")


# The page's own tokens follow the theme Streamlit renders (not the OS), so a switched theme stays consistent.
DARK_TOKENS = (":root{--cw-paper:#0D1230;--cw-paper-2:#151B45;--cw-ink:#E7E9F7;--cw-primary:#8E9BFF;"
               "--cw-stamp:#FF7A70;--cw-sage:#6FCF9F;--cw-amber:#F2A94A;--cw-rule:#2A3166}")
_SUN = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">'
        '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2'
        'M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>')
_MOON = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">'
         '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>')
PAGES = ("", "ask", "checklist", "gapcheck", "evaluation")


def theme_switch() -> None:
    """Light/dark switch in the header. Streamlit keeps the viewer's theme in localStorage under
    stActiveTheme-<path>-v2 (one key per first-loaded path); the button writes it for every page and reloads.
    Static script, no user input, so unsafe_allow_javascript carries no injection risk."""
    target = "Light" if is_dark() else "Dark"
    pages = ",".join(f"'{p}'" for p in PAGES)
    st.html(f"""<button class="cw-theme-switch" id="cw-theme-switch" type="button"
  title="Switch to the {target.lower()} theme (reloads the page)">{_SUN if target == "Light" else _MOON}{target}</button>
<script>
(() => {{
  const b = document.getElementById("cw-theme-switch");
  if (!b) return;
  b.onclick = () => {{
    const base = location.pathname.replace(/(ask|checklist|gapcheck|evaluation)[/]?$/, "");
    for (const p of [{pages}]) {{
      try {{ localStorage.setItem(`stActiveTheme-${{base}}${{p}}-v2`, JSON.stringify("{target}")); }} catch (e) {{}}
    }}
    location.reload();
  }};
}})();
</script>""", width="content", unsafe_allow_javascript=True)


def _config(name: str, default: Optional[str] = None) -> Optional[str]:
    """Streamlit secret, else environment variable, else default.

    Accessing st.secrets raises when no secrets.toml exists, so env-only
    deployments must not touch it unguarded.
    """
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: BLE001 - StreamlitSecretNotFoundError / parse errors
        pass
    return os.getenv(name, default)


def backend_url() -> str:
    return str(_config("BACKEND_BASE_URL", "http://localhost:8000")).rstrip("/")


def api_key() -> Optional[str]:
    return _config("API_KEY")


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001
        return False


def backend_state() -> str:
    """'ready' | 'waking' | 'down', probed once per session until ready."""
    s = st.session_state
    if s.get("backend_state") != "ready":
        state, latency = probe_health(backend_url(), timeout=1.5)
        if state == "waking":
            s.setdefault("backend_since", time.time())
        s.backend_state, s.backend_latency = state, latency
    return s.backend_state


@st.cache_data(ttl=600, show_spinner=False)
def _facts(url: str, key: Optional[str]) -> list:
    return get_facts(url, key)


def facts_or_none() -> Optional[list]:
    """facts.yaml labels for wording a profile; None (plain wording) when the API is not ready or has no checklist.
    Failures are not cached, so labels appear once a sleeping API wakes."""
    if backend_state() != "ready":
        return None
    try:
        return _facts(backend_url(), api_key())
    except httpx.HTTPError:
        return None


def note(text: str, icon: str) -> None:
    """A neutral, bordered note in the theme's own ink (Streamlit's info blue is not in this palette)."""
    with st.container(border=True, width=PROSE_WIDTH):
        st.markdown(f"{icon} {text}")


@st.fragment(run_every=2)
def wake_watcher() -> None:
    """While the API boots: show progress, re-probe every 2 s, reload the app when ready."""
    s = st.session_state
    elapsed = time.time() - s.get("backend_since", time.time())
    if elapsed > WAKE_LIMIT_S:
        st.error(
            "The API did not come up within 3 minutes. It may be paused or failing to build: "
            "check the Hugging Face Space status page, then reload this page.",
            icon=":material/cloud_off:",
        )
        return
    # Draw progress first: the probe below blocks for up to its timeout while the host boots.
    st.markdown(f":orange-badge[:material/hourglass_top: Waking the API · {elapsed:.0f} s]")
    st.caption(
        "The free host puts the API to sleep after 48 hours without traffic. "
        "This first request restarts it, which usually takes under a minute."
    )
    state, latency = probe_health(backend_url(), timeout=2.0)
    if state == "ready":
        s.backend_state, s.backend_latency = "ready", latency
        st.rerun(scope="app")


def status_badge() -> None:
    """Compact API status for the page header."""
    state = backend_state()
    if state == "ready":
        latency = st.session_state.backend_latency
        shown = f"{latency * 1000:.0f} ms" if latency < 1 else f"{latency:.1f} s"
        st.markdown(f":green-badge[:material/check_circle: API ready · {shown}]")
    elif state == "waking":
        st.markdown(":orange-badge[:material/hourglass_top: API waking up]")
    else:
        st.markdown(f":red-badge[:material/error: API unreachable at {backend_url()}]")
