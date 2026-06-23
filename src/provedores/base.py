"""Contrato comum para provedores de LLM (offline-first)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RespostaLLM:
    """Resultado padronizado de uma chamada ao LLM."""
    resposta: str
    tempo_ms: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    erro: bool = False


class LLMProvider(ABC):
    """Interface mínima que todo provider deve cumprir."""
    nome: str = "base"

    @abstractmethod
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
        ...

    @abstractmethod
    def modelo_disponivel(self, modelo: str) -> bool:
        ...

    @abstractmethod
    def warmup(
        self,
        modelos: dict[str, str],
        funcoes: tuple[str, ...] = ("rapido",),
        keep_alive: str = "5m",
    ) -> None:
        ...
