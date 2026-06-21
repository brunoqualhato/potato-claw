"""
Coordenador: roteia perguntas para o agente correto.
Estratégias:
1. Roteamento rápido por palavras-chave (sem LLM)
2. Roteamento inteligente via LLM (fallback)
"""

import re

from src.core.config import AGENTES, COORDENADOR_MODELO, COORDENADOR_SYSTEM
from src.core.llm import chamar_coordenador


def rotear_por_palavras_chave(texto: str) -> str | None:
    """
    Tenta rotear usando palavras-chave (sem LLM).
    Retorna None se não tiver confiança suficiente.
    """
    texto_lower = texto.lower()
    tokens = set(re.findall(r"[a-zA-Z0-9_\-]{2,}", texto_lower))
    pontuacao: dict[str, int] = {}

    for nome, agente in AGENTES.items():
        score = 0
        for palavra in agente["palavras_chave"]:
            palavra_lower = palavra.lower()

            # Palavra composta precisa bater como substring.
            if " " in palavra_lower and palavra_lower in texto_lower:
                score += 2
                continue

            # Palavra simples precisa casar token para evitar falso positivo.
            if palavra_lower in tokens:
                score += 2
            elif palavra_lower in texto_lower:
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


def rotear_por_llm(texto: str) -> str:
    """Usa o LLM coordenador para classificar a pergunta."""
    resposta = chamar_coordenador(texto, COORDENADOR_MODELO, COORDENADOR_SYSTEM)

    for nome in AGENTES:
        if nome in resposta:
            return nome

    return "analista"


def rotear(texto: str) -> str:
    """
    Estratégia híbrida:
    1. Palavras-chave (instantâneo)
    2. LLM coordenador (fallback)
    """
    agente = rotear_por_palavras_chave(texto)
    if agente:
        return agente

    return rotear_por_llm(texto)
