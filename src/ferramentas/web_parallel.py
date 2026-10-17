"""Parallel Search MCP opcional, anônimo, via SDK oficial e Streamable HTTP."""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from urllib.parse import urlsplit

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "potato-claw/0.1.0 (+https://github.com/brunoqualhato/potato-claw)"
MAX_TEXTO = 6000


def _dados(resultado) -> dict:
    if resultado.isError:
        mensagem = " ".join(c.text for c in resultado.content if c.type == "text")
        raise RuntimeError(f"Parallel Search MCP: {mensagem}")
    dados = resultado.structuredContent
    if dados is None:
        textos = [c.text for c in resultado.content if c.type == "text"]
        if len(textos) != 1:
            raise ValueError("Resposta Parallel sem conteúdo JSON")
        dados = json.loads(textos[0])
    if not isinstance(dados, dict) or not isinstance(dados.get("results"), list):
        raise ValueError("Resposta Parallel inválida: results ausente")
    return dados


def _documentos(dados: dict, limite: int) -> list[dict]:
    documentos = []
    for r in dados["results"]:
        if not isinstance(r, dict):
            raise ValueError("Resultado Parallel inválido")
        url = r.get("url", "")
        if not isinstance(url, str):
            continue
        partes = urlsplit(url)
        if partes.scheme not in {"http", "https"} or not partes.hostname or partes.username or partes.password:
            continue
        trechos = r.get("excerpts", [])
        if not isinstance(trechos, list) or any(not isinstance(t, str) for t in trechos):
            raise ValueError("Trechos Parallel inválidos")
        texto = "\n".join(trechos)[:MAX_TEXTO]
        if texto.strip():
            documentos.append({"url": url, "titulo": r.get("title") or url, "markdown": texto})
        if len(documentos) >= limite:
            break
    return documentos


async def _buscar(query: str, limite: int, buscar_conteudo: bool, timeout: float) -> list[dict]:
    # Não importa MCP/HTTP até o usuário escolher este backend.
    try:
        import anyio
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from mcp.types import Implementation
    except ImportError as e:
        raise RuntimeError("Instale Parallel com: pip install -r requirements-parallel.txt") from e

    # Cliente próprio anônimo, sem auth/cookies. Conserva proxies e certificados
    # do ambiente; HTTPX só usa netrc quando NetRCAuth é configurado explicitamente.
    with anyio.fail_after(timeout):
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=timeout,
        ) as http:
            async with streamable_http_client(MCP_URL, http_client=http) as (leitura, escrita, _):
                async with ClientSession(
                    leitura, escrita, read_timeout_seconds=timedelta(seconds=timeout),
                    client_info=Implementation(name="potato-claw", version="0.1.0"),
                ) as sessao:
                    await sessao.initialize()
                    ferramentas = {f.name for f in (await sessao.list_tools()).tools}
                    exigidas = {"web_search", "web_fetch"} if buscar_conteudo else {"web_search"}
                    if not exigidas <= ferramentas:
                        raise RuntimeError("Parallel MCP não oferece as ferramentas necessárias")
                    dados = _dados(await sessao.call_tool(
                        "web_search", {"objective": query, "search_queries": [query]},
                    ))
                    documentos = _documentos(dados, limite)
                    if buscar_conteudo and documentos:
                        argumentos = {
                            "urls": [d["url"] for d in documentos],
                            "objective": query[:200], "search_queries": [query],
                        }
                        if dados.get("session_id"):
                            argumentos["session_id"] = dados["session_id"]
                        paginas = _dados(await sessao.call_tool("web_fetch", argumentos))
                        # Falhas parciais de fetch conservam os trechos da busca e suas fontes.
                        por_url = {d["url"]: d for d in _documentos(paginas, limite)}
                        documentos = [por_url.get(d["url"], d) for d in documentos]
                    return documentos


def buscar_documentos_parallel(query: str, max_resultados: int = 5, *, buscar_conteudo: bool = False) -> list[dict]:
    """Busca trechos citáveis; opcionalmente extrai páginas para o web RAG profundo."""
    if not query.strip() or max_resultados <= 0:
        return []
    timeout = float(os.environ.get("NEURON_PARALLEL_TIMEOUT", "20"))
    if not 0 < timeout <= 120:
        raise ValueError("NEURON_PARALLEL_TIMEOUT deve ser maior que 0 e no máximo 120 segundos")
    limite = min(max_resultados, 20)

    def executar():
        try:
            return asyncio.run(_buscar(query, limite, buscar_conteudo, timeout))
        except ExceptionGroup as grupo:
            # Task groups do transporte encapsulam erros HTTP/MCP. Conserva a
            # causa útil (inclusive quota) para o chamador e o log da aplicação.
            causa = grupo
            while isinstance(causa, ExceptionGroup):
                causa = causa.exceptions[0]
            raise causa from grupo

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return executar()
    # A API também pode chamar o pipeline síncrono dentro de um event loop.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="potato-parallel") as pool:
        return pool.submit(executar).result()
