"""
Ferramenta de pesquisa web via DuckDuckGo, Tavily ou Parallel MCP.
"""

from ddgs import DDGS

from src.ferramentas.web_provider import obter_provedor_web


def pesquisar_web(query: str, max_resultados: int = 5) -> str:
    """Pesquisa usando a seleção compartilhada dos pipelines simples e profundo."""
    try:
        provedor = obter_provedor_web()
        if provedor == "none":
            return "Pesquisa web desativada."
        if provedor == "tavily":
            from src.ferramentas.web_tavily import pesquisar_web_tavily
            return pesquisar_web_tavily(query, max_resultados)
        if provedor == "parallel":
            from src.ferramentas.web_parallel import buscar_documentos_parallel
            documentos = buscar_documentos_parallel(query, max_resultados)
            if not documentos:
                return "Nenhum resultado encontrado."
            return f"Resultados para: '{query}'\n\n" + "\n\n".join(
                f"{i}. {d['titulo']}\n   {d['markdown']}\n   Fonte: {d['url']}"
                for i, d in enumerate(documentos, 1)
            )
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
