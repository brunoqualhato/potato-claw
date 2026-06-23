from unittest.mock import patch
from src.provedores.base import RespostaLLM


@patch("src.core.llm._provider")
def test_chamar_llm_retorna_dict_compativel(mock_provider):
    mock_provider.chat.return_value = RespostaLLM(
        resposta="ok", tempo_ms=12, tokens_entrada=5, tokens_saida=2
    )
    from src.core import llm
    out = llm.chamar_llm("m", "sys", [{"role": "user", "content": "oi"}], stream=False)
    assert out == {"resposta": "ok", "tempo_ms": 12, "tokens_entrada": 5, "tokens_saida": 2}


@patch("src.core.llm._provider")
def test_verificar_modelo_disponivel_delega(mock_provider):
    mock_provider.modelo_disponivel.return_value = True
    from src.core import llm
    assert llm.verificar_modelo_disponivel("qwen3:1.7b") is True
    mock_provider.modelo_disponivel.assert_called_once_with("qwen3:1.7b")
