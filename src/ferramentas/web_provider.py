"""Seleção compartilhada de busca, sem importar SDKs opcionais."""

import os


def obter_provedor_web() -> str:
    """Mantém Tavily configurado/DuckDuckGo, salvo escolha explícita."""
    escolha = os.environ.get("NEURON_WEB_PROVIDER", "").strip().lower()
    if escolha:
        if escolha not in {"duckduckgo", "tavily", "parallel", "none"}:
            raise ValueError("NEURON_WEB_PROVIDER deve ser duckduckgo, tavily, parallel ou none")
        return escolha
    return "tavily" if os.environ.get("TAVILY_API_KEY") else "duckduckgo"
