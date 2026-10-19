import pytest
from app.services.regs import DATA_DIR, load_provisions
from app.services.regs.registry import Registry

def test_every_committed_rule_quote_occurs_in_its_basis_provision():
    path = DATA_DIR / "regs-v1.jsonl"
    if not path.exists():
        pytest.skip("corpus not built")
    reg = Registry.load(DATA_DIR / "applicability.yaml", load_provisions(path))
    assert reg.unverified_quotes() == []
