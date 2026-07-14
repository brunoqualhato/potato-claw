"""
Ferramenta de pesquisa web via Tavily.

Alternativa ao DuckDuckGo — ativada quando TAVILY_API_KEY está definida.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tavily import TavilyClient

logger = logging.getLogger(__name__)

# Inicializado sob demanda para evitar erro se a chave não existir.
_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient()  # usa TAVILY_API_KEY do ambiente
    return _client


def pesquisar_web_tavily(query: str, max_resultados: int = 5) -> str:
    """Pesquisa na web usando Tavily. Retorna texto formatado igual ao DuckDuckGo."""
    try:
        response = _get_client().search(
            query=query,
            max_results=max_resultados,
            search_depth="basic",
        )

        resultados = response.get("results", [])
        if not resultados:
            return "Nenhum resultado encontrado."

        texto = f"Resultados para: '{query}'\n\n"
        for i, r in enumerate(resultados, 1):
            texto += f"{i}. {r.get('title', '')}\n"
            texto += f"   {r.get('content', '')}\n"
            texto += f"   Fonte: {r.get('url', '')}\n\n"

        return texto

    except Exception as e:
        return f"Erro na pesquisa: {str(e)}"


@dataclass(slots=True)
class ResultadoBuscaTavily:
    titulo: str
    url: str
    snippet: str


def buscar_urls_tavily(query: str, max_resultados: int = 5) -> list[ResultadoBuscaTavily]:
    """Retorna URLs rankeadas do Tavily (compatível com ResultadoBusca)."""
    try:
        response = _get_client().search(
            query=query,
            max_results=max_resultados,
            search_depth="basic",
        )

        return [
            ResultadoBuscaTavily(
                titulo=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
            )
            for r in response.get("results", [])
            if r.get("url")
        ]
    except Exception as e:
        logger.warning("Erro na busca Tavily: %s", e)
        return []
