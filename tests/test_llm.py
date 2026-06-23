"""Testes da fachada de providers."""

from unittest.mock import MagicMock

from src.core import llm
from src.provedores.base import RespostaLLM


def test_chamar_llm_preserva_formato_publico(monkeypatch):
    provider = MagicMock()
    provider.chat.return_value = RespostaLLM(
        resposta="ok",
        tempo_ms=12,
        tokens_entrada=3,
        tokens_saida=2,
    )
    monkeypatch.setattr(llm, "_provider", provider)

    resultado = llm.chamar_llm("modelo", "system", [], stream=False)

    assert resultado == {
        "resposta": "ok",
        "tempo_ms": 12,
        "tokens_entrada": 3,
        "tokens_saida": 2,
        "erro": False,
    }


def test_warmup_repassa_funcoes_selecionadas(monkeypatch):
    provider = MagicMock()
    monkeypatch.setattr(llm, "_provider", provider)
    modelos = {"rapido": "modelo-rapido", "embedding": "modelo-embedding"}

    llm.warmup_modelos(modelos, funcoes=("embedding",))

    provider.warmup.assert_called_once_with(
        modelos,
        funcoes=("embedding",),
        keep_alive=llm.KEEP_ALIVE_PRINCIPAL,
    )
