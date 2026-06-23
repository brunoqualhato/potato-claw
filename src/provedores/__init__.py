"""Camada de abstração de provedores de LLM."""
from src.provedores import ollama_provider  # noqa: F401  (auto-registra "ollama")

try:
    from src.provedores import litellm_provider  # noqa: F401  (registra "litellm" se possível)
except Exception:
    pass
