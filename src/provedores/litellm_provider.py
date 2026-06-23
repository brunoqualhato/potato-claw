"""Provider OPCIONAL via litellm (100+ backends, inclui cloud).

Desligado por padrão. litellm é importado de forma lazy: o módulo carrega
sem a dependência instalada; só falha se alguém chamar de fato o provider.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from importlib.util import find_spec

from src.provedores.base import LLMProvider, RespostaLLM


class LiteLLMProvider(LLMProvider):
    nome = "litellm"

    def _lib(self):
        import litellm  # lazy: só quando usado
        return litellm

    def chat(
        self,
        modelo: str,
        system_prompt: str,
        mensagens: list[dict],
        *,
        stream: bool = True,
        max_tokens: int = 2048,
        temperatura: float = 0.7,
        num_ctx: int | None = None,
        num_thread: int | None = None,
        keep_alive: str | None = None,
        timeout: float | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> RespostaLLM:
        litellm = self._lib()
        msgs = [{"role": "system", "content": system_prompt}, *mensagens]
        inicio = time.time()
        resp = litellm.completion(
            model=modelo,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperatura,
            timeout=timeout,
        )
        texto = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {}) or {}
        return RespostaLLM(
            resposta=texto,
            tempo_ms=int((time.time() - inicio) * 1000),
            tokens_entrada=usage.get("prompt_tokens", 0),
            tokens_saida=usage.get("completion_tokens", 0),
        )

    def modelo_disponivel(self, modelo: str) -> bool:
        # litellm não tem inventário local; assume-se configurado externamente.
        return True

    def warmup(
        self,
        modelos: dict[str, str],
        funcoes: tuple[str, ...] = ("rapido",),
        keep_alive: str = "5m",
    ) -> None:
        return None


try:
    from src.provedores import registry as _registry

    if find_spec("litellm") is not None:
        _registry.registrar("litellm", LiteLLMProvider)
except Exception:
    pass
