"""Isolamento de memoria por sessao (canal, pessoa) para multi-user."""
from unittest.mock import MagicMock

from src.agentes.executor import SistemaAgentes
from src.memoria.cache import Cache
from src.memoria.sqlite import Memoria


def test_sessoes_nao_misturam_historico(tmp_path):
    m = Memoria(arquivo=str(tmp_path / "m.db"))
    m.sessao_ativa = "telegram:1"
    m.salvar_mensagem("user", "sou da sessao 1")
    m.sessao_ativa = "telegram:2"
    m.salvar_mensagem("user", "sou da sessao 2")

    assert [x["content"] for x in m.ultimas_mensagens(10)] == ["sou da sessao 2"]
    m.sessao_ativa = "telegram:1"
    assert [x["content"] for x in m.ultimas_mensagens(10)] == ["sou da sessao 1"]
    m.fechar()


def test_resumo_isolado_por_sessao(tmp_path):
    m = Memoria(arquivo=str(tmp_path / "m.db"))
    m.sessao_ativa = "telegram:1"
    m.salvar_resumo("resumo da 1")
    m.sessao_ativa = "telegram:2"
    assert m.ultimo_resumo() is None
    m.sessao_ativa = "telegram:1"
    assert m.ultimo_resumo() == "resumo da 1"
    m.fechar()


def test_retrocompat_sessao_global(tmp_path):
    # Sem setar sessao_ativa (=""), comportamento global continua funcionando.
    m = Memoria(arquivo=str(tmp_path / "m.db"))
    m.salvar_mensagem("user", "global")
    assert [x["content"] for x in m.ultimas_mensagens(10)] == ["global"]
    m.fechar()


def test_cache_conversacional_e_isolado_por_sessao(tmp_path):
    memoria = Memoria(arquivo=str(tmp_path / "m.db"))
    cache = Cache(arquivo=str(tmp_path / "cache.json"))
    semantica = MagicMock()
    sistema = SistemaAgentes(memoria=memoria, cache=cache, semantica=semantica)

    memoria.sessao_ativa = "telegram:1"
    cache.salvar(sistema._cache_key("generalista", "meu segredo"), "resposta privada")

    memoria.sessao_ativa = "telegram:2"
    assert cache.buscar(sistema._cache_key("generalista", "meu segredo")) is None
    memoria.fechar()


def test_executor_propaga_sessao_para_memoria_semantica(tmp_path):
    memoria = Memoria(arquivo=str(tmp_path / "m.db"))
    cache = Cache(arquivo=str(tmp_path / "cache.json"))
    semantica = MagicMock()
    semantica.buscar_similar.return_value = []
    sistema = SistemaAgentes(memoria=memoria, cache=cache, semantica=semantica)
    memoria.sessao_ativa = "telegram:42"

    sistema._salvar("pergunta privada", "resposta privada", "generalista", 2, 0)

    semantica.adicionar.assert_called_once_with(
        "pergunta privada",
        "resposta privada",
        "generalista",
        metadata={"sessao": "telegram:42"},
    )
    memoria.fechar()
