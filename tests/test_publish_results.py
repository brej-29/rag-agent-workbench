"""Snapshots the UI reads (spec §17.3): latest run per kind, runs grouped by prompt, corpus coverage."""
import json

from eval import publish_results as pub


def _write(d, name, data):
    (d / name).write_text(json.dumps(data), encoding="utf-8")


def test_modes_keep_the_latest_run_at_k6_per_mode(tmp_path):
    _write(tmp_path, "regs_dense_20261020T000000Z.json", {"k": 6, "n": 46, "aggregate": {"recall": 0.83, "mrr": 0.75}})
    _write(tmp_path, "regs_dense_20261024T000000Z.json", {"k": 6, "n": 50, "aggregate": {"recall": 0.85, "mrr": 0.76}})
    _write(tmp_path, "regs_dense_20261025T000000Z.json", {"k": 10, "n": 50, "aggregate": {"recall": 0.9, "mrr": 0.76}})
    _write(tmp_path, "regs_diversified-boost_20261024T000000Z.json",
           {"k": 6, "n": 50, "aggregate": {"recall": 0.87, "mrr": 0.84}})
    assert pub.regs_modes(tmp_path) == {"dense": {"n": 50, "recall": 0.85, "mrr": 0.76},
                                        "diversified-boost": {"n": 50, "recall": 0.87, "mrr": 0.84}}
    assert pub.regs_retrieval(tmp_path) is None  # no plain diversified run yet


def test_gapcheck_snapshot_groups_runs_of_the_latest_prompt(tmp_path):
    base = {"accuracy": 0.8, "verified_rate": 1.0, "missing_caught": 5, "missing_planted": 6, "tokens": 11000,
            "false_covered": []}
    _write(tmp_path, "gapcheck_20261024T010000Z.json", {**base, "accuracy": 0.5})  # an older prompt: no hash
    _write(tmp_path, "gapcheck_20261025T010000Z.json", {**base, "prompt_sha": "abc", "accuracy": 0.76})
    _write(tmp_path, "gapcheck_20261025T020000Z.json", {**base, "prompt_sha": "abc", "accuracy": 0.88, "rows": []})
    snap = pub.gapcheck_runs(tmp_path)
    assert snap["accuracy"] == 0.88 and [r["accuracy"] for r in snap["runs"]] == [0.76, 0.88]
    assert snap["runs"][0] == {"run": "20261025T010000Z", "accuracy": 0.76, "verified_rate": 1.0, "missing_caught": 5,
                               "missing_planted": 6, "tokens": 11000, "false_covered": 0}


def test_corpus_summary_counts_provisions_per_instrument(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps([{"key": "gdpr", "url": "u", "retrieved": "2026-10-20"}]),
                                            encoding="utf-8")
    (tmp_path / "regs-v1.jsonl").write_text(
        "\n".join(json.dumps({"id": f"gdpr/art-{i}", "binding": "regulation"}) for i in range(3)), encoding="utf-8")
    assert pub.corpus_summary(tmp_path) == [{"key": "gdpr", "name": "GDPR", "binding": "regulation", "provisions": 3,
                                             "url": "u", "retrieved": "2026-10-20"}]


def test_publish_writes_the_compliance_snapshots(tmp_path):
    reports, results = tmp_path / "reports", tmp_path / "results"
    reports.mkdir()
    _write(reports, "router_20261020T040300Z.json", {"n": 36, "accuracy": 0.944, "confusion": {}, "misses": []})
    _write(reports, "regs_diversified_20261024T023549Z.json",
           {"mode": "diversified", "k": 6, "n": 50, "aggregate": {"recall": 0.87, "mrr": 0.85}, "per_query": []})
    done = pub.publish(reports, results, pub.REGS_DATA)
    assert {"router", "regs_retrieval", "regs_modes", "corpus"} <= set(done)
    corpus = json.loads((results / "corpus.json").read_text(encoding="utf-8"))
    assert {c["key"] for c in corpus} >= {"eu-ai-act", "gdpr", "nis2", "nis2-ir", "dora", "cra", "nist-csf"}
