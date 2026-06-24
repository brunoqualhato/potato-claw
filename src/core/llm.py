"""
Fachada de LLM. Delega ao provider resolvido pelo registry (default ollama)
e aplica o tuning de hardware fraco (num_ctx por nivel, keep_alive) ao delegar.
Mantém as assinaturas públicas usadas por executor.py e main.py.
"""
from __future__ import annotations

import ollama
from rich.console import Console

from src.core.config import (
    KEEP_ALIVE_EFEMERO,
    KEEP_ALIVE_PRINCIPAL,
    NUM_CTX_AUXILIAR,
)
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
    num_ctx: int | None = None,
    keep_alive: str | None = KEEP_ALIVE_PRINCIPAL,
) -> dict:
    r = _provider.chat(
        modelo, system_prompt, mensagens,
        stream=stream, max_tokens=max_tokens, temperatura=temperatura,
        num_ctx=num_ctx, keep_alive=keep_alive,
    )
    return {
        "resposta": r.resposta,
        "tempo_ms": r.tempo_ms,
        "tokens_entrada": r.tokens_entrada,
        "tokens_saida": r.tokens_saida,
    }


def chamar_coordenador(pergunta: str, modelo: str, system_prompt: str) -> str:
    # Modelo auxiliar: janela mínima + keep_alive efêmero (não ocupa RAM).
    r = _provider.chat(
        modelo, system_prompt, [{"role": "user", "content": pergunta}],
        stream=False, max_tokens=20, temperatura=0.1,
        num_ctx=NUM_CTX_AUXILIAR, keep_alive=KEEP_ALIVE_EFEMERO,
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
        num_ctx=NUM_CTX_AUXILIAR, keep_alive=KEEP_ALIVE_EFEMERO,
    )
    return r.resposta


def verificar_modelo_disponivel(modelo: str) -> bool:
    return _provider.modelo_disponivel(modelo)


def warmup_modelos(
    modelos: dict[str, str],
    funcoes: tuple[str, ...] = ("rapido",),
    keep_alive: str = KEEP_ALIVE_PRINCIPAL,
):
    """
    Pré-carrega modelos na RAM do Ollama ao iniciar.
    Evita latência de ~2-3s na primeira chamada.

    Args:
        modelos: dict de perfil MODELOS (rapido, completo, etc.)
        funcoes: funções que devem ser aquecidas; embedding usa a API própria
        keep_alive: tempo para manter na RAM
    """
    selecionados = [(funcao, modelos[funcao]) for funcao in funcoes if funcao in modelos]
    vistos: set[str] = set()
    for funcao, modelo in selecionados:
        if modelo in vistos:
            continue
        vistos.add(modelo)
        try:
            if funcao == "embedding":
                ollama.embeddings(model=modelo, prompt="warmup")
            else:
                ollama.chat(
                    model=modelo,
                    messages=[{"role": "user", "content": "oi"}],
                    options={"num_predict": 1, "num_ctx": NUM_CTX_AUXILIAR},
                    keep_alive=keep_alive,
                )
            console.print(f"  [dim]🔥 {modelo} carregado[/dim]")
        except Exception:
            pass
