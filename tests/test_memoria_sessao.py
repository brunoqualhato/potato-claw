"""Isolamento de memoria por sessao (canal, pessoa) para multi-user."""
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
