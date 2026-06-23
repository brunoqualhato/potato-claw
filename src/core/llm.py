"""
Interface com o Ollama.
Gerencia chamadas ao LLM com controle de nível de performance.

Tuning para hardware fraco (ver src/core/config.py):
- num_ctx / num_thread aplicados em toda chamada (RAM e CPU sob controle);
- keep_alive efêmero solta da RAM os modelos auxiliares (coordenador/resumo);
- timeout duro evita travar o processo quando o modelo engasga sob swap;
- streaming desacoplado da CLI via callback on_token (a API pode consumir sem
  poluir o stdout do servidor).
"""

import time
from typing import Callable, Optional

import ollama
from rich.console import Console

from src.core.config import (
    KEEP_ALIVE_EFEMERO,
    KEEP_ALIVE_PRINCIPAL,
    NUM_CTX_AUXILIAR,
    NUM_THREAD,
    OLLAMA_TIMEOUT,
)

console = Console()

# Client único com timeout duro. Reutilizado em todas as chamadas para não
# recriar conexões e para garantir que nenhuma chamada pendure o processo.
_client = ollama.Client(timeout=OLLAMA_TIMEOUT)


def _montar_options(
    temperatura: float,
    max_tokens: int,
    num_ctx: Optional[int],
) -> dict:
    """Options comuns a todas as chamadas, com tuning de hardware fraco."""
    options = {
        "temperature": temperatura,
        "num_predict": max_tokens,
        "num_thread": NUM_THREAD,
    }
    # num_ctx só entra quando definido; 0/None deixa o Ollama decidir.
    if num_ctx:
        options["num_ctx"] = num_ctx
    return options


def chamar_llm(
    modelo: str,
    system_prompt: str,
    mensagens: list[dict],
    stream: bool = True,
    max_tokens: int = 2048,
    temperatura: float = 0.7,
    num_ctx: Optional[int] = None,
    keep_alive: Optional[str] = KEEP_ALIVE_PRINCIPAL,
    on_token: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Chama o Ollama com streaming.
    Retorna dict com {resposta, tempo_ms, tokens_entrada, tokens_saida, erro}.

    on_token: se fornecido, recebe cada pedaço do streaming (a API usa isso para
    streamar ao cliente). Se None, mantém o comportamento da CLI (imprime no
    console). Assim a camada de LLM não fica acoplada ao stdout.
    """
    msgs = [{"role": "system", "content": system_prompt}]
    for m in mensagens:
        msgs.append({"role": m["role"], "content": m["content"]})

    inicio = time.time()
    resposta_completa = ""
    tokens_in = 0
    tokens_out = 0
    erro = False

    options = _montar_options(temperatura, max_tokens, num_ctx)

    if stream:
        try:
            stream_response = _client.chat(
                model=modelo,
                messages=msgs,
                stream=True,
                options=options,
                keep_alive=keep_alive,
            )
            for chunk in stream_response:
                texto = chunk["message"]["content"]
                resposta_completa += texto

                if on_token is not None:
                    on_token(texto)
                else:
                    console.print(texto, end="", style="green")

                if "eval_count" in chunk:
                    tokens_out = chunk["eval_count"]
                if "prompt_eval_count" in chunk:
                    tokens_in = chunk["prompt_eval_count"]

                # Detecção de degeneração: para se o modelo repetir o mesmo bloco.
                # Evita desperdiçar tempo/recursos quando o modelo entra em loop,
                # justamente o objetivo do tuning para hardware fraco.
                if len(resposta_completa) > 300 and resposta_completa[-150:] == resposta_completa[-300:-150]:
                    if on_token is None:
                        console.print("\n[dim]⚠️ Repetição detectada, parando.[/dim]")
                    break

            if on_token is None:
                console.print()
        except Exception as e:
            erro = True
            resposta_completa = f"Erro ao chamar modelo {modelo}: {str(e)}"
            console.print(f"\n[red]{resposta_completa}[/red]")
    else:
        try:
            response = _client.chat(
                model=modelo,
                messages=msgs,
                options=options,
                keep_alive=keep_alive,
            )
            resposta_completa = response["message"]["content"]
            tokens_in = response.get("prompt_eval_count", 0)
            tokens_out = response.get("eval_count", 0)
        except Exception as e:
            erro = True
            resposta_completa = f"Erro ao chamar modelo {modelo}: {str(e)}"

    tempo_ms = int((time.time() - inicio) * 1000)

    return {
        "resposta": resposta_completa,
        "tempo_ms": tempo_ms,
        "tokens_entrada": tokens_in,
        "tokens_saida": tokens_out,
        "erro": erro,
    }


def chamar_coordenador(pergunta: str, modelo: str, system_prompt: str) -> str:
    """
    Chama o coordenador para classificar a pergunta.
    Sem streaming - precisa da resposta completa.
    Modelo auxiliar: janela mínima e keep_alive efêmero para não ocupar RAM.
    """
    try:
        response = _client.chat(
            model=modelo,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pergunta},
            ],
            options={
                "temperature": 0.1,
                "num_predict": 20,
                "num_thread": NUM_THREAD,
                "num_ctx": NUM_CTX_AUXILIAR,
            },
            keep_alive=KEEP_ALIVE_EFEMERO,
        )
        return response["message"]["content"].strip().lower()
    except Exception as e:
        console.print(f"[red]Erro no coordenador: {e}[/red]")
        return "analista"


def resumir_conversa(modelo: str, mensagens: list[dict]) -> str:
    """Gera resumo compacto da conversa. Chamada auxiliar (efêmera)."""
    texto_conversa = "\n".join(
        f"{m['role']}: {m['content']}" for m in mensagens
    )

    try:
        response = _client.chat(
            model=modelo,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Resuma a conversa abaixo em no máximo 2 frases. "
                        "Capture o objetivo principal do usuário e decisões tomadas."
                    ),
                },
                {"role": "user", "content": texto_conversa},
            ],
            options={
                "temperature": 0.3,
                "num_predict": 100,
                "num_thread": NUM_THREAD,
                "num_ctx": NUM_CTX_AUXILIAR,
            },
            keep_alive=KEEP_ALIVE_EFEMERO,
        )
        return response["message"]["content"]
    except Exception:
        return ""


def verificar_modelo_disponivel(modelo: str) -> bool:
    """Verifica se o modelo está disponível localmente."""
    try:
        resposta = _client.list()

        nomes_raw: list[str] = []

        # Formato antigo/JSON-like
        if isinstance(resposta, dict):
            for m in resposta.get("models", []):
                if isinstance(m, dict):
                    nomes_raw.append(m.get("name") or m.get("model") or "")

        # Formato atual da lib: ListResponse com lista de objetos Model
        elif hasattr(resposta, "models"):
            for m in getattr(resposta, "models", []):
                nomes_raw.append(getattr(m, "model", "") or getattr(m, "name", ""))

        # Normaliza para evitar falso negativo por case/tag.
        nomes = {n.strip().lower() for n in nomes_raw if n}
        nomes_base = {n.split(":", 1)[0] for n in nomes}

        alvo = modelo.strip().lower()
        alvo_base = alvo.split(":", 1)[0]

        # Se o modelo solicitado já tem tag, não aceitar apenas o nome-base.
        if ":" in alvo:
            return alvo in nomes

        return alvo in nomes or f"{alvo}:latest" in nomes or alvo_base in nomes_base
    except Exception:
        return False


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
