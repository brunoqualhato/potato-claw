from unittest.mock import MagicMock, patch

from src.provedores.base import RespostaLLM
from src.provedores.ollama_provider import OllamaProvider


@patch("src.provedores.ollama_provider.ollama")
def test_chat_sem_stream_retorna_resposta(mock_ollama):
    mock_ollama.Client.return_value.chat.return_value = {
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
    mock_ollama.Client.return_value.chat.return_value = {"message": {"content": "x"}}
    p = OllamaProvider()
    p.chat("m", "s", [{"role": "user", "content": "oi"}], stream=False,
           num_ctx=1024, num_thread=4, keep_alive="0s")
    _, kwargs = mock_ollama.Client.return_value.chat.call_args
    assert kwargs["options"]["num_ctx"] == 1024
    assert kwargs["options"]["num_thread"] == 4
    assert kwargs["keep_alive"] == "0s"


@patch("src.provedores.ollama_provider.ollama")
def test_chat_aplica_num_thread_padrao_quando_ausente(mock_ollama):
    from src.core.config import NUM_THREAD

    mock_ollama.Client.return_value.chat.return_value = {"message": {"content": "x"}}
    p = OllamaProvider()
    p.chat("m", "s", [{"role": "user", "content": "oi"}], stream=False)
    _, kwargs = mock_ollama.Client.return_value.chat.call_args
    assert kwargs["options"]["num_thread"] == NUM_THREAD


@patch("src.provedores.ollama_provider.ollama")
def test_chat_aplica_timeout_padrao_e_customizado(mock_ollama):
    from src.core.config import OLLAMA_TIMEOUT

    mock_ollama.Client.return_value.chat.return_value = {"message": {"content": "x"}}
    p = OllamaProvider()

    p.chat("m", "s", [{"role": "user", "content": "oi"}], stream=False)
    assert mock_ollama.Client.call_args.kwargs["timeout"] == OLLAMA_TIMEOUT

    p.chat("m", "s", [{"role": "user", "content": "oi"}], stream=False, timeout=5)
    assert mock_ollama.Client.call_args.kwargs["timeout"] == 5


@patch("src.provedores.ollama_provider.ollama")
def test_modelo_disponivel_lista(mock_ollama):
    resp = MagicMock()
    resp.models = [MagicMock(model="qwen3:1.7b")]
    mock_ollama.list.return_value = resp
    p = OllamaProvider()
    assert p.modelo_disponivel("qwen3:1.7b") is True
    assert p.modelo_disponivel("inexistente") is False
