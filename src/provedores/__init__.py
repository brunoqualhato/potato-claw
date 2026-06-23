"""Camada de abstração de provedores de LLM."""
from src.provedores import (
    litellm_provider,  # noqa: F401
    ollama_provider,  # noqa: F401  (auto-registra "ollama")
)
