"""Provider padrão: Ollama local. Preserva o tuning para PC fraco."""
from __future__ import annotations

import time

import ollama
from rich.console import Console

from src.provedores.base import LLMProvider, RespostaLLM

console = Console()


class OllamaProvider(LLMProvider):
    nome = "ollama"

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
    ) -> RespostaLLM:
        msgs = [{"role": "system", "content": system_prompt}]
        for m in mensagens:
            msgs.append({"role": m["role"], "content": m["content"]})

        options: dict = {"temperature": temperatura, "num_predict": max_tokens}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if num_thread is not None:
            options["num_thread"] = num_thread

        extra: dict = {}
        if keep_alive is not None:
            extra["keep_alive"] = keep_alive

        inicio = time.time()
        resposta = ""
        tokens_in = 0
        tokens_out = 0

        try:
            if stream:
                for chunk in ollama.chat(
                    model=modelo, messages=msgs, stream=True, options=options, **extra
                ):
                    texto = chunk["message"]["content"]
                    resposta += texto
                    console.print(texto, end="", style="green")
                    if "eval_count" in chunk:
                        tokens_out = chunk["eval_count"]
                    if "prompt_eval_count" in chunk:
                        tokens_in = chunk["prompt_eval_count"]
                    if len(resposta) > 300 and resposta[-150:] == resposta[-300:-150]:
                        console.print("\n[dim]⚠️ Repetição detectada, parando.[/dim]")
                        break
                console.print()
            else:
                r = ollama.chat(model=modelo, messages=msgs, options=options, **extra)
                resposta = r["message"]["content"]
                tokens_in = r.get("prompt_eval_count", 0)
                tokens_out = r.get("eval_count", 0)
        except Exception as e:
            resposta = f"Erro ao chamar modelo {modelo}: {e}"
            console.print(f"\n[red]{resposta}[/red]")

        return RespostaLLM(
            resposta=resposta,
            tempo_ms=int((time.time() - inicio) * 1000),
            tokens_entrada=tokens_in,
            tokens_saida=tokens_out,
        )

    def modelo_disponivel(self, modelo: str) -> bool:
        try:
            resposta = ollama.list()
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

    def warmup(self, modelos: dict[str, str], keep_alive: str = "10m") -> None:
        for modelo in set(modelos.values()):
            try:
                ollama.chat(
                    model=modelo,
                    messages=[{"role": "user", "content": "oi"}],
                    options={"num_predict": 1},
                    keep_alive=keep_alive,
                )
                console.print(f"  [dim]🔥 {modelo} carregado[/dim]")
            except Exception:
                pass
