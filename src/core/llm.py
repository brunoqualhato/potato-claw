"""Fachada compatível para provedores de LLM."""

from collections.abc import Callable

from src.core.config import (
    KEEP_ALIVE_EFEMERO,
    KEEP_ALIVE_PRINCIPAL,
    NUM_CTX_AUXILIAR,
    NUM_THREAD,
    OLLAMA_TIMEOUT,
)
from src.provedores import registry
from src.provedores.base import LLMProvider

_provider: LLMProvider = registry.criar("ollama")


def configurar_provider(nome: str, **kwargs) -> LLMProvider:
    """Troca explicitamente o provider usado pela fachada."""
    global _provider
    _provider = registry.criar(nome, **kwargs)
    return _provider


def provider_atual() -> LLMProvider:
    return _provider


def chamar_llm(
    modelo: str,
    system_prompt: str,
    mensagens: list[dict],
    stream: bool = True,
    max_tokens: int = 2048,
    temperatura: float = 0.7,
    num_ctx: int | None = None,
    keep_alive: str | None = KEEP_ALIVE_PRINCIPAL,
    on_token: Callable[[str], None] | None = None,
) -> dict:
    resultado = _provider.chat(
        modelo,
        system_prompt,
        mensagens,
        stream=stream,
        max_tokens=max_tokens,
        temperatura=temperatura,
        num_ctx=num_ctx,
        num_thread=NUM_THREAD,
        keep_alive=keep_alive,
        timeout=OLLAMA_TIMEOUT,
        on_token=on_token,
    )
    return {
        "resposta": resultado.resposta,
        "tempo_ms": resultado.tempo_ms,
        "tokens_entrada": resultado.tokens_entrada,
        "tokens_saida": resultado.tokens_saida,
        "erro": resultado.erro,
    }


def chamar_coordenador(pergunta: str, modelo: str, system_prompt: str) -> str:
    resultado = _provider.chat(
        modelo,
        system_prompt,
        [{"role": "user", "content": pergunta}],
        stream=False,
        max_tokens=20,
        temperatura=0.1,
        num_ctx=NUM_CTX_AUXILIAR,
        num_thread=NUM_THREAD,
        keep_alive=KEEP_ALIVE_EFEMERO,
        timeout=OLLAMA_TIMEOUT,
    )
    if resultado.erro:
        return "analista"
    return resultado.resposta.strip().lower() or "analista"


def resumir_conversa(modelo: str, mensagens: list[dict]) -> str:
    texto = "\n".join(f"{m['role']}: {m['content']}" for m in mensagens)
    resultado = _provider.chat(
        modelo,
        (
            "Resuma a conversa abaixo em no máximo 2 frases. "
            "Capture o objetivo principal do usuário e decisões tomadas."
        ),
        [{"role": "user", "content": texto}],
        stream=False,
        max_tokens=100,
        temperatura=0.3,
        num_ctx=NUM_CTX_AUXILIAR,
        num_thread=NUM_THREAD,
        keep_alive=KEEP_ALIVE_EFEMERO,
        timeout=OLLAMA_TIMEOUT,
    )
    return "" if resultado.erro else resultado.resposta


def verificar_modelo_disponivel(modelo: str) -> bool:
    return _provider.modelo_disponivel(modelo)


def warmup_modelos(
    modelos: dict[str, str],
    funcoes: tuple[str, ...] = ("rapido",),
    keep_alive: str = KEEP_ALIVE_PRINCIPAL,
) -> None:
    _provider.warmup(modelos, funcoes=funcoes, keep_alive=keep_alive)
