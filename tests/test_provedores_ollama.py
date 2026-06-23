from unittest.mock import MagicMock

from src.provedores.base import RespostaLLM
from src.provedores.ollama_provider import OllamaProvider


def test_chat_sem_stream_retorna_resposta():
    client = MagicMock()
    client.chat.return_value = {
        "message": {"content": "resposta teste"},
        "prompt_eval_count": 7,
        "eval_count": 3,
    }
    p = OllamaProvider(client=client)
    r = p.chat("modelo-x", "sys", [{"role": "user", "content": "oi"}], stream=False)
    assert isinstance(r, RespostaLLM)
    assert r.resposta == "resposta teste"
    assert r.tokens_entrada == 7
    assert r.tokens_saida == 3


def test_chat_passa_num_ctx_e_num_thread_nas_options():
    client = MagicMock()
    client.chat.return_value = {"message": {"content": "x"}}
    p = OllamaProvider(client=client)
    p.chat(
        "m",
        "s",
        [{"role": "user", "content": "oi"}],
        stream=False,
        num_ctx=1024,
        num_thread=4,
        keep_alive="0s",
    )
    _, kwargs = client.chat.call_args
    assert kwargs["options"]["num_ctx"] == 1024
    assert kwargs["options"]["num_thread"] == 4
    assert kwargs["keep_alive"] == "0s"


def test_modelo_disponivel_lista():
    client = MagicMock()
    resp = MagicMock()
    resp.models = [MagicMock(model="qwen3:1.7b")]
    client.list.return_value = resp
    p = OllamaProvider(client=client)
    assert p.modelo_disponivel("qwen3:1.7b") is True
    assert p.modelo_disponivel("inexistente") is False
