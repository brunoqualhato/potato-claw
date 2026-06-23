import pytest

from src.provedores.base import LLMProvider, RespostaLLM


def test_resposta_llm_tem_defaults():
    r = RespostaLLM(resposta="oi")
    assert r.resposta == "oi"
    assert r.tempo_ms == 0
    assert r.tokens_entrada == 0
    assert r.tokens_saida == 0


def test_llmprovider_nao_instancia_direto():
    with pytest.raises(TypeError):
        LLMProvider()  # ABC com métodos abstratos
