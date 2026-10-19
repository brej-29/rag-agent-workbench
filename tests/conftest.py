"""Make the suite hermetic: never read a developer's .env, never trace to LangSmith.

Runs before any test module imports app.core.config.
"""
import os

os.environ["DISABLE_DOTENV"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
