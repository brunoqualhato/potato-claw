"""Provider padrão: Ollama local. Preserva o tuning para PC fraco."""
from __future__ import annotations

import time
from collections.abc import Callable

import ollama
from rich.console import Console

from src.core.config import NUM_CTX_AUXILIAR, NUM_THREAD, OLLAMA_TIMEOUT
from src.provedores.base import LLMProvider, RespostaLLM

console = Console()


class OllamaProvider(LLMProvider):
    nome = "ollama"

    def __init__(self, client=None) -> None:
        self._client = client or ollama.Client(timeout=OLLAMA_TIMEOUT)

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
        msgs = [{"role": "system", "content": system_prompt}]
        for m in mensagens:
            msgs.append({"role": m["role"], "content": m["content"]})

        options: dict = {
            "temperature": temperatura,
            "num_predict": max_tokens,
            "num_thread": num_thread or NUM_THREAD,
        }
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        extra: dict = {}
        if keep_alive is not None:
            extra["keep_alive"] = keep_alive

        inicio = time.time()
        resposta = ""
        tokens_in = 0
        tokens_out = 0
        erro = False

        try:
            if stream:
                for chunk in self._client.chat(
                    model=modelo, messages=msgs, stream=True, options=options, **extra
                ):
                    texto = chunk["message"]["content"]
                    resposta += texto
                    if on_token is not None:
                        on_token(texto)
                    else:
                        console.print(texto, end="", style="green")
                    if "eval_count" in chunk:
                        tokens_out = chunk["eval_count"]
                    if "prompt_eval_count" in chunk:
                        tokens_in = chunk["prompt_eval_count"]
                    if len(resposta) > 300 and resposta[-150:] == resposta[-300:-150]:
                        if on_token is None:
                            console.print("\n[dim]⚠️ Repetição detectada, parando.[/dim]")
                        break
                if on_token is None:
                    console.print()
            else:
                r = self._client.chat(model=modelo, messages=msgs, options=options, **extra)
                resposta = r["message"]["content"]
                tokens_in = r.get("prompt_eval_count", 0)
                tokens_out = r.get("eval_count", 0)
        except Exception as e:
            erro = True
            resposta = f"Erro ao chamar modelo {modelo}: {e}"
            console.print(f"\n[red]{resposta}[/red]")

        return RespostaLLM(
            resposta=resposta,
            tempo_ms=int((time.time() - inicio) * 1000),
            tokens_entrada=tokens_in,
            tokens_saida=tokens_out,
            erro=erro,
        )

    def modelo_disponivel(self, modelo: str) -> bool:
        try:
            resposta = self._client.list()
            nomes_raw: list[str] = []
            if isinstance(resposta, dict):
                for m in resposta.get("models", []):
                    if isinstance(m, dict):
                        nomes_raw.append(m.get("name") or m.get("model") or "")
            elif hasattr(resposta, "models"):
                for m in getattr(resposta, "models", []):
                    nomes_raw.append(getattr(m, "model", "") or getattr(m, "name", ""))
            nomes = {n.strip().lower() for n in nomes_raw if n}
            nomes_base = {n.split(":", 1)[0] for n in nomes}
            alvo = modelo.strip().lower()
            alvo_base = alvo.split(":", 1)[0]
            if ":" in alvo:
                return alvo in nomes
            return alvo in nomes or f"{alvo}:latest" in nomes or alvo_base in nomes_base
        except Exception:
            return False

    def warmup(
        self,
        modelos: dict[str, str],
        funcoes: tuple[str, ...] = ("rapido",),
        keep_alive: str = "5m",
    ) -> None:
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
                    self._client.chat(
                        model=modelo,
                        messages=[{"role": "user", "content": "oi"}],
                        options={
                            "num_predict": 1,
                            "num_ctx": NUM_CTX_AUXILIAR,
                            "num_thread": NUM_THREAD,
                        },
                        keep_alive=keep_alive,
                    )
                console.print(f"  [dim]🔥 {modelo} carregado[/dim]")
            except Exception:
                pass


from src.provedores import registry as _registry  # noqa: E402

_registry.registrar("ollama", OllamaProvider)
