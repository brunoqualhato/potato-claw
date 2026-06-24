"""Testes da interface HTTP sem iniciar servidor real."""

import asyncio
from unittest.mock import MagicMock

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
