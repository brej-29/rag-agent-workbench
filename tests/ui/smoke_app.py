"""UI boot smoke test (run with the UI lock, not the backend lock):
    PYTHONPATH=frontend python tests/ui/smoke_app.py
Renders both pages headlessly with env-only config, no secrets.toml and no backend."""
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

os.environ.setdefault("BACKEND_BASE_URL", "http://127.0.0.1:9")
os.environ.setdefault("API_KEY", "smoke")
APP = str(Path(__file__).resolve().parents[2] / "frontend" / "app.py")

at = AppTest.from_file(APP, default_timeout=60).run()
assert not at.exception, at.exception
assert at.chat_input, "chat input not rendered"
print("Ask page OK:", at.title[0].value if at.title else "(no title)")

at.switch_page("views/evaluation.py").run()
assert not at.exception, at.exception
headers = [h.value for h in at.header]
assert {"Retrieval", "Abstention", "Routing", "Checklist", "Gap check"} <= set(headers), headers
assert len(at.tabs) == 2, [t.label for t in at.tabs]
print("Evaluation page OK:", headers)

import tempfile

import services.eval_data as eval_data

saved, eval_data.RESULTS_DIR = eval_data.RESULTS_DIR, Path(tempfile.mkdtemp())  # no snapshots at all
at.switch_page("views/evaluation.py").run()
assert not at.exception, at.exception
assert any("Not published yet" in c.value for c in at.caption)
eval_data.RESULTS_DIR = saved
print("Evaluation page OK with no snapshots")

at.switch_page("views/checklist.py").run()
assert not at.exception, at.exception
assert at.title and at.title[0].value == "Your checklist", [t.value for t in at.title]
print("Checklist page OK (API unreachable path)")


at.switch_page("views/gapcheck.py").run()
assert not at.exception, at.exception
assert at.title and at.title[0].value == "Gap check"
print("Gap check page OK (API unreachable path)")
