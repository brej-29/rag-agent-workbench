"""Make the suite hermetic: never read a developer's .env, never trace to LangSmith.

Runs before any test module imports app.core.config.
"""
import os

os.environ["DISABLE_DOTENV"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

# Required Settings fields: obvious dummies so full-graph tests can build Settings.
# No test reaches the network; Pinecone and the LLM are always faked.
for _name, _value in {"PINECONE_API_KEY": "test-key-not-real", "PINECONE_INDEX_NAME": "test-index",
                      "PINECONE_HOST": "https://test.invalid"}.items():
    os.environ.setdefault(_name, _value)
