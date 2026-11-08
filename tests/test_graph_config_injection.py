import warnings

from app.services.chat import graph


def test_every_node_receives_runnable_config():
    """LangGraph silently skips config injection (and only warns) when a node's
    `config` annotation is unrecognised, e.g. a string under
    `from __future__ import annotations`.  Tracing then never reaches the node."""
    graph._graph = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*'config' parameter should be typed.*")
            compiled = graph.get_chat_graph()
    finally:
        graph._graph = None
    for name, node in compiled.builder.nodes.items():
        if "config" in node.runnable.func.__code__.co_varnames[: node.runnable.func.__code__.co_argcount]:
            assert "config" in node.runnable.func_accepts, name
