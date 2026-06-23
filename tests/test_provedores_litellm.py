import sys
from unittest.mock import MagicMock, patch
from src.provedores.litellm_provider import LiteLLMProvider
from src.provedores.base import RespostaLLM


def test_chat_usa_litellm_lazy():
    fake = MagicMock()
    fake.completion.return_value = {
        "choices": [{"message": {"content": "oi litellm"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    with patch.dict(sys.modules, {"litellm": fake}):
        p = LiteLLMProvider()
        r = p.chat("ollama/qwen3:1.7b", "sys", [{"role": "user", "content": "x"}], stream=False)
    assert isinstance(r, RespostaLLM)
    assert r.resposta == "oi litellm"
    assert r.tokens_entrada == 4
    assert r.tokens_saida == 2
