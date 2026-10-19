"""App-wide state shared by every page: backend address, key, and wake-up status."""
from __future__ import annotations

import os
import time
from typing import Optional

import streamlit as st

from services.backend_client import probe_health

# The free host can take a while to boot a sleeping Space; give up after this.
WAKE_LIMIT_S = 180


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
