"""MCP real sobre transporte HTTP determinístico; nenhuma LLM/rede externa."""

import asyncio
import builtins
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from src.ferramentas import web, web_parallel, web_rag
from src.ferramentas.web_provider import obter_provedor_web

URL = "https://docs.python.org/3/whatsnew/3.14.html"
TRECHO = "Python 3.14 inclui novos recursos de desenvolvimento."


@pytest.fixture(autouse=True)
def ambiente_limpo(monkeypatch):
    for nome in ("NEURON_WEB_PROVIDER", "NEURON_PARALLEL_TIMEOUT", "TAVILY_API_KEY", "PARALLEL_API_KEY"):
        monkeypatch.delenv(nome, raising=False)


@pytest.fixture
def servidor_mcp(monkeypatch):
    """Observa requisições do SDK, sem substituir descoberta, sessão ou dispatcher."""
    estado = SimpleNamespace(pedidos=[], clientes=[], instancias=[], modo="normal", sse=False)
    cliente_original = httpx.AsyncClient

    async def responder(request):
        estado.pedidos.append(request)
        if request.method != "POST":
            return httpx.Response(405)
        pedido = json.loads(request.content)
        metodo = pedido["method"]
        if metodo.startswith("notifications/"):
            return httpx.Response(202)
        if metodo == "initialize":
            resultado = {
                "protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture-parallel", "version": "1"},
            }
        elif metodo == "tools/list":
            nomes = [] if estado.modo == "missing_tool" else ["web_search", "web_fetch"]
            resultado = {"tools": [{"name": nome, "inputSchema": {"type": "object"}} for nome in nomes]}
        else:
            assert metodo == "tools/call"
            nome = pedido["params"]["name"]
            if estado.modo == "timeout":
                await asyncio.sleep(10)
            if estado.modo == "http_error":
                return httpx.Response(503, text="temporarily unavailable")
            if estado.modo == "rpc_error":
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": pedido["id"],
                    "error": {"code": -32603, "message": "upstream failed"},
                })
            dados = {
                "results": [{"url": URL, "title": "Python 3.14", "excerpts": [
                    "Conteúdo extraído da página." if nome == "web_fetch" else TRECHO,
                ]}],
                "session_id": "fixture-session-from-search", "errors": [],
            }
            if estado.modo == "unsafe_urls":
                dados["results"] = [
                    {"url": url, "excerpts": ["inválido"]}
                    for url in ["file:///etc/passwd", "javascript:alert(1)", "https://user:pass@example.com"]
                ] + dados["results"]
            if estado.modo == "empty" or (estado.modo == "partial_fetch" and nome == "web_fetch"):
                dados["results"] = []
            resultado = {"content": [{"type": "text", "text": json.dumps(dados)}]}
            if estado.modo != "text_only":
                resultado["structuredContent"] = dados
            if estado.modo == "malformed":
                resultado = {"content": [{"type": "text", "text": "not json"}]}
            if estado.modo == "tool_error":
                resultado = {"isError": True, "content": [{"type": "text", "text": "quota exceeded"}]}
        resposta = {"jsonrpc": "2.0", "id": pedido["id"], "result": resultado}
        if estado.sse:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=(
                "event: message\ndata: " + json.dumps(resposta) + "\n\n"
            ).encode())
        return httpx.Response(200, json=resposta)

    class Cliente(cliente_original):
        def __init__(self, **kwargs):
            estado.clientes.append(kwargs)
            super().__init__(**kwargs, transport=httpx.MockTransport(responder))
            estado.instancias.append(self)

    monkeypatch.setattr(httpx, "AsyncClient", Cliente)
    monkeypatch.setenv("NEURON_WEB_PROVIDER", "parallel")
    return estado


def chamadas(estado):
    return [json.loads(r.content) for r in estado.pedidos if r.method == "POST"]


@pytest.mark.parametrize("sse", [False, True])
def test_busca_simples_anonima_pelo_sdk(servidor_mcp, monkeypatch, tmp_path, sse):
    servidor_mcp.sse = sse
    monkeypatch.setenv("PARALLEL_API_KEY", "nao-utilizar")
    monkeypatch.setenv("TAVILY_API_KEY", "escolha-explicita-parallel-prevalece")
    netrc = tmp_path / "credentials.netrc"
    netrc.write_text("machine search.parallel.ai login usuario password nao-utilizar\n")
    monkeypatch.setenv("NETRC", str(netrc))
    resposta = web.pesquisar_web("Python novidades versão", max_resultados=1)
    assert TRECHO in resposta and URL in resposta
    pedidos = chamadas(servidor_mcp)
    assert [p["method"] for p in pedidos] == [
        "initialize", "notifications/initialized", "tools/list", "tools/call",
    ]
    assert pedidos[-1]["params"] == {
        "name": "web_search",
        "arguments": {"objective": "Python novidades versão", "search_queries": ["Python novidades versão"]},
    }
    assert "auth" not in servidor_mcp.clientes[0]
    for req in servidor_mcp.pedidos:
        assert str(req.url) == "https://search.parallel.ai/mcp"
        assert req.headers["user-agent"] == web_parallel.USER_AGENT
        assert "authorization" not in req.headers and "cookie" not in req.headers


def test_grounding_rapido_e_profundo(servidor_mcp, monkeypatch):
    rapido = web_rag.pesquisar_web_rapida("Python novidades versão", 1)
    assert TRECHO in rapido and URL in rapido
    assert not any(p.get("params", {}).get("name") == "web_fetch" for p in chamadas(servidor_mcp))
    extrator = MagicMock(return_value={"erro": False, "resposta": "Fato extraído com fonte."})
    monkeypatch.setattr(web_rag, "chamar_llm", extrator)
    profundo = web_rag.pesquisar_web_profunda("Python novidades versão", 1)
    assert "Fato extraído" in profundo and URL in profundo
    assert "Conteúdo extraído da página." in extrator.call_args.kwargs["mensagens"][0]["content"]
    fetch = [p for p in chamadas(servidor_mcp) if p.get("params", {}).get("name") == "web_fetch"][0]
    assert fetch["params"]["arguments"] == {
        "urls": [URL], "objective": "Python novidades versão", "search_queries": ["Python novidades versão"],
        "session_id": "fixture-session-from-search",
    }
    assert all(r.headers["user-agent"] == web_parallel.USER_AGENT for r in servidor_mcp.pedidos)
    assert all(str(r.url) == web_parallel.MCP_URL for r in servidor_mcp.pedidos)


@pytest.mark.parametrize("modo", ["text_only", "partial_fetch", "unsafe_urls"])
def test_variantes_de_conteudo(servidor_mcp, modo):
    servidor_mcp.modo = modo
    documentos = web_parallel.buscar_documentos_parallel("Python novidades versão", 1, buscar_conteudo=True)
    assert len(documentos) == 1 and documentos[0]["url"] == URL
    if modo == "partial_fetch":
        assert documentos[0]["markdown"] == TRECHO


@pytest.mark.parametrize("modo", ["tool_error", "malformed", "missing_tool", "http_error", "rpc_error", "empty"])
def test_falhas_nao_trocam_provedor(servidor_mcp, monkeypatch, modo):
    servidor_mcp.modo = modo
    duck = MagicMock(side_effect=AssertionError("Não trocar para DuckDuckGo"))
    monkeypatch.setattr(web, "DDGS", duck)
    monkeypatch.setattr(web_rag, "DDGS", duck)
    assert web_rag.pesquisar_web_rapida("Python novidades versão") == ""
    simples = web.pesquisar_web("Python novidades versão")
    assert "Erro na pesquisa" in simples or "Nenhum resultado" in simples
    if modo == "tool_error":
        assert "quota exceeded" in simples
    duck.assert_not_called()


def test_deadline_total(servidor_mcp, monkeypatch):
    servidor_mcp.modo = "timeout"
    monkeypatch.setenv("NEURON_PARALLEL_TIMEOUT", "0.05")
    inicio = time.monotonic()
    with pytest.raises(TimeoutError):
        web_parallel.buscar_documentos_parallel("Python novidades versão")
    assert time.monotonic() - inicio < 1
    assert all(c.is_closed for c in servidor_mcp.instancias)


def test_cancelamento_async_fecha_transporte(servidor_mcp):
    servidor_mcp.modo = "timeout"

    async def executar():
        tarefa = asyncio.create_task(web_parallel._buscar("Python novidades versão", 1, False, 10))
        await asyncio.sleep(0.03)
        tarefa.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tarefa

    asyncio.run(executar())
    assert all(c.is_closed for c in servidor_mcp.instancias)


def test_chamador_com_event_loop_ativo(servidor_mcp):
    async def executar():
        return web.pesquisar_web("Python novidades versão")

    assert TRECHO in asyncio.run(executar())


@pytest.mark.parametrize("timeout", ["nan", "inf", "0", "-1", "121", "invalid"])
def test_timeout_invalido_sem_rede(servidor_mcp, monkeypatch, timeout):
    monkeypatch.setenv("NEURON_PARALLEL_TIMEOUT", timeout)
    with pytest.raises(ValueError):
        web_parallel.buscar_documentos_parallel("Python novidades versão")
    assert not servidor_mcp.pedidos


def test_limites_e_entrada_vazia(servidor_mcp):
    assert web_parallel.buscar_documentos_parallel("  ") == []
    assert web_parallel.buscar_documentos_parallel("consulta", 0) == []
    assert not servidor_mcp.pedidos
    assert len(web_parallel.buscar_documentos_parallel("consulta", 999)) == 1


def test_default_sem_chave_e_escolhas_existentes(monkeypatch):
    assert obter_provedor_web() == "duckduckgo"
    monkeypatch.setenv("TAVILY_API_KEY", "chave-existente")
    assert obter_provedor_web() == "tavily"
    monkeypatch.setenv("NEURON_WEB_PROVIDER", "duckduckgo")
    assert obter_provedor_web() == "duckduckgo"
    monkeypatch.setenv("NEURON_WEB_PROVIDER", "")
    assert obter_provedor_web() == "tavily"


def test_sdk_opcional_ausente_erro_acionavel(monkeypatch):
    importar = builtins.__import__

    def sem_mcp(nome, *args, **kwargs):
        if nome == "mcp":
            raise ImportError("SDK opcional ausente")
        return importar(nome, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", sem_mcp)
    monkeypatch.setenv("NEURON_WEB_PROVIDER", "parallel")
    assert "pip install -r requirements-parallel.txt" in web.pesquisar_web("consulta")


def test_dispatch_incumbentes(monkeypatch):
    duck = MagicMock()
    duck.return_value.__enter__.return_value.text.return_value = [
        {"title": "Página", "href": URL, "body": TRECHO},
    ]
    monkeypatch.setattr(web, "DDGS", duck)
    monkeypatch.setattr(web_rag, "DDGS", duck)
    assert TRECHO in web.pesquisar_web("consulta")
    assert web_rag.buscar_urls("consulta")[0].url == URL
    from src.ferramentas import web_tavily
    tavily = MagicMock(return_value="resultado Tavily")
    monkeypatch.setattr(web_tavily, "pesquisar_web_tavily", tavily)
    monkeypatch.setenv("TAVILY_API_KEY", "chave-existente")
    assert web.pesquisar_web("consulta") == "resultado Tavily"
    monkeypatch.setenv("NEURON_WEB_PROVIDER", "duckduckgo")
    assert TRECHO in web.pesquisar_web("consulta")


def test_desativado_sem_rede(servidor_mcp, monkeypatch):
    monkeypatch.setenv("NEURON_WEB_PROVIDER", "none")
    monkeypatch.setenv("TAVILY_API_KEY", "chave-existente")
    assert web.pesquisar_web("consulta") == "Pesquisa web desativada."
    assert web_rag.buscar_urls("consulta") == []
    assert web_rag.pesquisar_web_profunda("consulta") == ""
    assert web_rag.pesquisar_web_rapida("consulta") == ""
    assert not servidor_mcp.pedidos


def test_escolha_invalida_sem_fallback(servidor_mcp, monkeypatch):
    monkeypatch.setenv("NEURON_WEB_PROVIDER", "paralell")
    assert "NEURON_WEB_PROVIDER" in web.pesquisar_web("consulta")
    assert web_rag.pesquisar_e_extrair("consulta").fatos_extraidos == ""
    assert web_rag.pesquisar_web_rapida("consulta") == ""
    assert web_rag.pesquisar_web_profunda("consulta") == ""
    assert not servidor_mcp.pedidos


def test_executor_consume_fontes_e_preserva_permissoes(servidor_mcp, monkeypatch, tmp_path):
    from src.agentes import executor
    from src.core.analisador import IntencaoAnalisada
    from src.memoria.cache import Cache
    from src.memoria.sqlite import Memoria

    semantica = MagicMock()
    semantica.buscar_similar.return_value = []
    sistema = executor.SistemaAgentes(
        memoria=Memoria(arquivo=str(tmp_path / "memoria.db")),
        cache=Cache(arquivo=str(tmp_path / "cache.json")), semantica=semantica,
    )
    modelo = MagicMock(return_value={
        "resposta": "Python 3.14 inclui recursos descritos na documentação oficial. " * 4,
        "tempo_ms": 1, "tokens_entrada": 1, "tokens_saida": 1, "erro": False,
    })
    monkeypatch.setattr(executor, "analisar_intencao", lambda _: IntencaoAnalisada(precisa_web=True))
    monkeypatch.setattr(executor, "chamar_llm", modelo)
    monkeypatch.setattr(executor, "AUTO_APROVAR_ACOES_LOCAIS", False)
    try:
        sistema.nivel_forcado = 2
        resposta = sistema.executar("generalista", "pesquise novidades recentes da linguagem Python")
        assert "Python 3.14" in resposta
        mensagens = modelo.call_args.kwargs["mensagens"]
        assert any(TRECHO in m["content"] and URL in m["content"] for m in mensagens)
        quantidade = len(servidor_mcp.pedidos)
        assert "Confirmação necessária" in sistema.executar("generalista", "criar arquivo teste.txt ::: conteúdo")
        assert not (tmp_path / "teste.txt").exists()
        assert len(servidor_mcp.pedidos) == quantidade
        monkeypatch.setattr(executor, "analisar_intencao", lambda _: IntencaoAnalisada(precisa_web=False))
        sistema.nivel_forcado = 2
        sistema.executar("generalista", "explique uma ideia sobre literatura")
        assert len(servidor_mcp.pedidos) == quantidade
    finally:
        sistema.fechar()
