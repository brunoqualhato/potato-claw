"""
Ferramenta de pesquisa web via DuckDuckGo (ou Tavily, quando configurado).
"""

import os

from ddgs import DDGS

_USA_TAVILY = bool(os.environ.get("TAVILY_API_KEY"))


def pesquisar_web(query: str, max_resultados: int = 5) -> str:
    """Pesquisa na web usando Tavily (se disponível) ou DuckDuckGo."""
    if _USA_TAVILY:
        from src.ferramentas.web_tavily import pesquisar_web_tavily
        return pesquisar_web_tavily(query, max_resultados)

    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(query, max_results=max_resultados))

        if not resultados:
            return "Nenhum resultado encontrado."

        texto = f"Resultados para: '{query}'\n\n"
        for i, r in enumerate(resultados, 1):
            texto += f"{i}. {r['title']}\n"
            texto += f"   {r['body']}\n"
            texto += f"   Fonte: {r['href']}\n\n"

        return texto

    except Exception as e:
        return f"Erro na pesquisa: {str(e)}"


def pesquisar_clima(cidade: str) -> str:
    """Pesquisa clima/temperatura de uma cidade com query otimizada."""
    query = f"temperatura clima agora {cidade} previsão do tempo hoje"
    return pesquisar_web(query, max_resultados=3)


def pesquisar_documentacao(tecnologia: str, versao: str = "") -> str:
    """Pesquisa documentação oficial de uma tecnologia/biblioteca."""
    versao_str = f" {versao}" if versao else ""
    query = f"{tecnologia}{versao_str} documentation official docs site:docs.* OR site:*.dev OR site:*.io"
    resultado = pesquisar_web(query, max_resultados=4)
    if "Nenhum resultado" in resultado or "Erro" in resultado:
        # Fallback sem restrição de domínio
        query_fallback = f"{tecnologia}{versao_str} documentação oficial tutorial"
        return pesquisar_web(query_fallback, max_resultados=4)
    return resultado


def pesquisar_cotacao(ativo: str) -> str:
    """Pesquisa cotação de um ativo financeiro ou criptomoeda."""
    query = f"cotação {ativo} hoje valor atual preço"
    return pesquisar_web(query, max_resultados=3)
