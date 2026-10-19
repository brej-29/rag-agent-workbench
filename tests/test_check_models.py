from scripts.check_models import deprecated_models

_HTML = """
<h3 id="a"><a href="#a">August 16, 2026: <code class="x">llama-3.1-8b-instant</code> and
<code class="x">llama-3.3-70b-versatile</code></a></h3>
<table><tr><td><code>llama-3.1-8b-instant</code></td><td><code>openai/gpt-oss-20b</code></td></tr></table>
<h3 id="b">Overview</h3>
"""


def test_only_heading_models_count_as_deprecated():
    found = deprecated_models(_HTML)
    assert set(found) == {"llama-3.1-8b-instant", "llama-3.3-70b-versatile"}
    assert found["llama-3.1-8b-instant"].startswith("August 16, 2026")
    assert "openai/gpt-oss-20b" not in found  # replacement in a table, not retired
