"""Testes da interface HTTP sem iniciar servidor real."""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import api


def test_chat_retorna_metadados_reais(monkeypatch):
    sistema = MagicMock()
    sistema.executar.return_value = "Resultado: 4"
    sistema.ultimo_agente = "generalista"
    sistema.ultimo_nivel = 1
    sistema.ultima_fonte = "ferramenta"
    monkeypatch.setattr(api, "_sistema", sistema)

    resposta = asyncio.run(api.chat(api.PerguntaRequest(pergunta="2+2"), None))

    assert resposta.resposta == "Resultado: 4"
    assert resposta.nivel == 1
    assert resposta.fonte == "ferramenta"


def test_chat_com_agente_preserva_agente_executado(monkeypatch):
    sistema = MagicMock()
    sistema.executar.return_value = "análise"
    sistema.ultimo_agente = "analista"
    sistema.ultimo_nivel = 3
    sistema.ultima_fonte = "llm_profundo"
    monkeypatch.setattr(api, "_sistema", sistema)

    req = api.PerguntaAgenteRequest(pergunta="analise", agente="analista")
    resposta = asyncio.run(api.chat_com_agente(req, None))

    assert resposta.agente == "analista"
    sistema.executar.assert_called_once_with("analista", "analise")


class TestAutenticacao:
    def test_sem_chave_configurada_libera_acesso(self, monkeypatch):
        monkeypatch.setattr(api, "_API_KEY", "")
        assert asyncio.run(api.verificar_api_key(None)) is None

    def test_chave_correta_autoriza(self, monkeypatch):
        monkeypatch.setattr(api, "_API_KEY", "segredo")
        assert asyncio.run(api.verificar_api_key("segredo")) == "segredo"

    def test_chave_errada_rejeita(self, monkeypatch):
        monkeypatch.setattr(api, "_API_KEY", "segredo")
        with pytest.raises(HTTPException) as exc:
            asyncio.run(api.verificar_api_key("errada"))
        assert exc.value.status_code == 401

    def test_chave_ausente_rejeita_quando_configurada(self, monkeypatch):
        monkeypatch.setattr(api, "_API_KEY", "segredo")
        with pytest.raises(HTTPException) as exc:
            asyncio.run(api.verificar_api_key(None))
        assert exc.value.status_code == 401


def test_stats_le_sob_lock(monkeypatch):
    """Comprova que estatisticas() roda sob o _lock e fora do event loop
    (via to_thread): captura o estado do lock e a thread no momento da leitura."""
    import threading

    capturado = {}
    main_thread = threading.get_ident()

    def estatisticas():
        capturado["lock_ativo"] = api._lock.locked()
        capturado["thread"] = threading.get_ident()
        return {"total": 1}

    sistema = MagicMock()
    sistema.estatisticas.side_effect = estatisticas
    monkeypatch.setattr(api, "_sistema", sistema)

    resultado = asyncio.run(api.stats(None))

    assert resultado == {"total": 1}
    sistema.estatisticas.assert_called_once()
    # ponto central da correção: leitura serializada pelo _lock, em worker thread.
    assert capturado["lock_ativo"] is True
    assert capturado["thread"] != main_thread
