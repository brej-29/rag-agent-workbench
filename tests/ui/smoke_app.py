"""UI boot smoke test (run with the UI lock, not the backend lock):
    PYTHONPATH=frontend python tests/ui/smoke_app.py
Renders frontend/app.py headlessly with env-only config and no secrets.toml."""
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

os.environ.setdefault("BACKEND_BASE_URL", "http://127.0.0.1:9")
os.environ.setdefault("API_KEY", "smoke")
at = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "frontend" / "app.py"), default_timeout=60).run()
assert not at.exception, at.exception
assert at.chat_input, "chat input not rendered"
print("UI smoke OK:", at.title[0].value if at.title else "(no title)")
