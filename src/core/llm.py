"""
Fachada de LLM. Delega ao provider resolvido pelo registry (default ollama).
Mantém as assinaturas públicas usadas por executor.py e main.py.
"""
from __future__ import annotations

from rich.console import Console

from src.provedores import registry

console = Console()

# Provider único do processo. Trocável via NEURON_PROVIDER no futuro.
_provider = registry.criar("ollama")


def chamar_llm(
    modelo: str,
    system_prompt: str,
    mensagens: list[dict],
    stream: bool = True,
    max_tokens: int = 2048,
    temperatura: float = 0.7,
) -> dict:
    r = _provider.chat(
        modelo, system_prompt, mensagens,
        stream=stream, max_tokens=max_tokens, temperatura=temperatura,
    )
    return {
        "resposta": r.resposta,
        "tempo_ms": r.tempo_ms,
        "tokens_entrada": r.tokens_entrada,
        "tokens_saida": r.tokens_saida,
    }


def chamar_coordenador(pergunta: str, modelo: str, system_prompt: str) -> str:
    r = _provider.chat(
        modelo, system_prompt, [{"role": "user", "content": pergunta}],
        stream=False, max_tokens=20, temperatura=0.1,
    )
    texto = r.resposta.strip().lower()
    return texto or "analista"


def resumir_conversa(modelo: str, mensagens: list[dict]) -> str:
    texto_conversa = "\n".join(f"{m['role']}: {m['content']}" for m in mensagens)
    r = _provider.chat(
        modelo,
        (
            "Resuma a conversa abaixo em no máximo 2 frases. "
            "Capture o objetivo principal do usuário e decisões tomadas."
        ),
        [{"role": "user", "content": texto_conversa}],
        stream=False, max_tokens=100, temperatura=0.3,
    )
    return r.resposta


def verificar_modelo_disponivel(modelo: str) -> bool:
    return _provider.modelo_disponivel(modelo)


def warmup_modelos(modelos: dict[str, str], keep_alive: str = "10m") -> None:
    _provider.warmup(modelos, keep_alive=keep_alive)
