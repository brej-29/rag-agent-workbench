"""Retrieved chunks carry the Pinecone doc_id so citations can be scored against
human-labelled evidence documents (eval/run_answers.py).  Internal state only:
the public SourceHit schema maps fields explicitly and is unchanged."""
from unittest.mock import MagicMock, patch

from app.services.chat.graph import corrective_retrieve, retrieve_context


def _settings():
    s = MagicMock()
    s.RAG_RERANK_ENABLED = False
    s.PINECONE_TEXT_FIELD = "chunk_text"
    s.RAG_CRAG_ENABLED = True
    s.RAG_CRAG_MAX_ITERS = 1
    s.RAG_CRAG_GOOD_SCORE = 0.9
    s.RAG_DEFAULT_TOP_K = 5
    return s


HIT = {"_id": "abc:0", "_score": 0.6, "fields": {"chunk_text": "t", "doc_id": "abc", "title": "T", "source": "wiki"}}


def test_retrieve_context_keeps_doc_id():
    state = {"query": "q", "namespace": "ns", "top_k": 5, "timings": {}}
    with patch("app.services.chat.graph.get_settings", return_value=_settings()), \
         patch("app.services.chat.graph.pinecone_search", return_value=[HIT]):
        result = retrieve_context(state)
    assert result["retrieved"][0]["doc_id"] == "abc"


def test_crag_re_retrieval_keeps_doc_id():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="better query")
    state = {"query": "q", "namespace": "ns", "top_k": 5, "top_score": 0.1, "retrieved": [], "timings": {}}
    with patch("app.services.chat.graph.get_settings", return_value=_settings()), \
         patch("app.services.chat.graph.get_llm", return_value=llm), \
         patch("app.services.chat.graph.pinecone_search", return_value=[HIT]):
        result = corrective_retrieve(state)
    assert result["retrieved"][0]["doc_id"] == "abc"


def test_chat_response_sources_expose_doc_id():
    from app.schemas.chat import SourceHit

    assert SourceHit(source="wiki", title="t", chunk_text="c", doc_id="abc").doc_id == "abc"
    assert SourceHit(source="web", title="t", chunk_text="c").doc_id == ""  # web results have none
