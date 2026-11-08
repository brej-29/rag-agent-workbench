from unittest.mock import MagicMock, patch

from app.services import pinecone_store


def _search_with(raw_hit):
    response = MagicMock()
    response.to_dict.return_value = {"result": {"hits": [raw_hit]}}
    index = MagicMock()
    index.search.return_value = response
    with patch.object(pinecone_store, "get_index", return_value=index):
        return pinecone_store.search(namespace="eval", query_text="q", top_k=1)


def test_pinecone_v8_plus_attribute_keys_are_normalised():
    (hit,) = _search_with({"id_": "doc:0", "score_": 0.55, "fields": {"doc_id": "doc"}})
    assert hit == {"_id": "doc:0", "_score": 0.55, "fields": {"doc_id": "doc"}}


def test_wire_keys_pass_through_unchanged():
    raw = {"_id": "doc:1", "_score": 0.3, "fields": {}}
    assert _search_with(raw) == [raw]
