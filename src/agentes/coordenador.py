"""
Coordenador: validação de prompts e roteamento por palavras-chave.

Nota: O roteamento principal agora é feito pelo analisador de intenção
(src/core/analisador.py) que usa LLM. Este módulo mantém:
  - Validação de prompt (sem LLM)
  - Roteamento por palavras-chave (usado como fallback rápido)
  - Detecção de baixo sinal
"""

import re

from src.core.config import AGENTES
from src.core.utils import normalizar

STOPWORDS_CURTAS = {
    "oi", "olá", "ola", "hello", "hey", "blz", "ok", "sim", "não", "nao", "valeu", "obrigado",
}

SAUDACOES_CURTAS = {"oi", "olá", "ola", "hello", "hey", "bom dia", "boa tarde", "boa noite"}


def _tokens(texto: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_\-]{2,}", normalizar(texto)))


def _baixo_sinal(texto: str) -> bool:
    """Detecta entradas curtas/aleatórias para usar agente generalista barato."""
    texto_limpo = texto.strip().lower()
    if not texto_limpo:
        return True

    tks = _tokens(texto_limpo)
    if texto_limpo in STOPWORDS_CURTAS:
        return True

    if len(tks) <= 2 and len(texto_limpo) <= 14:
        return True

    # Muito ruído (pouca semântica) também cai para generalista.
    apenas_simbolos = re.sub(r"[a-zA-Z0-9\s]", "", texto_limpo)
    if len(apenas_simbolos) >= max(3, len(texto_limpo) // 3):
        return True

    return False


def rotear_por_palavras_chave(texto: str) -> str | None:
    """
    Tenta rotear usando palavras-chave (sem LLM).
    Retorna None se não tiver confiança suficiente.
    """
    texto_norm = normalizar(texto)
    tokens = _tokens(texto)
    pontuacao: dict[str, int] = {}

    for nome, agente in AGENTES.items():
        if nome == "generalista":
            continue
        score = 0
        for palavra in agente["palavras_chave"]:
            palavra_norm = normalizar(palavra)

            # Palavra composta precisa bater como substring.
            if " " in palavra_norm and palavra_norm in texto_norm:
                score += 2
                continue

            # Palavra simples precisa casar token para evitar falso positivo.
            if palavra_norm in tokens:
                score += 2
            elif palavra_norm in texto_norm:
                score += 1

        if score > 0:
            pontuacao[nome] = score

    if not pontuacao:
        return None

    ranking = sorted(pontuacao.items(), key=lambda item: item[1], reverse=True)
    melhor, melhor_score = ranking[0]
    segundo_score = ranking[1][1] if len(ranking) > 1 else 0

    # Empate ou baixa confiança -> deixa para o LLM coordenador.
    if melhor_score < 2 or (melhor_score - segundo_score) <= 1:
        return None

    return melhor


def validar_prompt(texto: str) -> tuple[bool, str]:
    """
    Valida se o prompt tem sinal semântico mínimo para roteamento.
    Retorna (valido, motivo_ou_vazio).

    Esta é uma validação LEVE (sem LLM) — rejeita apenas input
    claramente insuficiente para processamento.
    """
    texto_limpo = texto.strip()

    if not texto_limpo:
        return False, "Sua mensagem veio vazia. Envie uma pergunta com mais contexto."

    if normalizar(texto_limpo) in SAUDACOES_CURTAS:
        return True, ""

    if len(texto_limpo) < 3:
        return False, "A mensagem está curta demais para roteamento. Descreva melhor sua tarefa."

    if _baixo_sinal(texto_limpo):
        return False, (
            "Não ficou claro o tipo de tarefa. "
            "Explique o objetivo em uma frase (ex.: programar, pesquisar ou analisar)."
        )

    return True, ""
