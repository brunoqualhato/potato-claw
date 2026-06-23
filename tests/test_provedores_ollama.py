from unittest.mock import patch, MagicMock
from src.provedores.ollama_provider import OllamaProvider
from src.provedores.base import RespostaLLM


@patch("src.provedores.ollama_provider.ollama")
def test_chat_sem_stream_retorna_resposta(mock_ollama):
    mock_ollama.chat.return_value = {
        "message": {"content": "resposta teste"},
        "prompt_eval_count": 7,
        "eval_count": 3,
    }
    p = OllamaProvider()
    r = p.chat("modelo-x", "sys", [{"role": "user", "content": "oi"}], stream=False)
    assert isinstance(r, RespostaLLM)
    assert r.resposta == "resposta teste"
    assert r.tokens_entrada == 7
    assert r.tokens_saida == 3


@patch("src.provedores.ollama_provider.ollama")
def test_chat_passa_num_ctx_e_num_thread_nas_options(mock_ollama):
    mock_ollama.chat.return_value = {"message": {"content": "x"}}
    p = OllamaProvider()
    p.chat("m", "s", [{"role": "user", "content": "oi"}], stream=False,
           num_ctx=1024, num_thread=4, keep_alive="0s")
    _, kwargs = mock_ollama.chat.call_args
    assert kwargs["options"]["num_ctx"] == 1024
    assert kwargs["options"]["num_thread"] == 4
    assert kwargs["keep_alive"] == "0s"


@patch("src.provedores.ollama_provider.ollama")
def test_modelo_disponivel_lista(mock_ollama):
    resp = MagicMock()
    resp.models = [MagicMock(model="qwen3:1.7b")]
    mock_ollama.list.return_value = resp
    p = OllamaProvider()
    assert p.modelo_disponivel("qwen3:1.7b") is True
    assert p.modelo_disponivel("inexistente") is False
