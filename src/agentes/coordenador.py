"""
Coordenador: roteia perguntas para o agente correto.
Estratégias:
1. Roteamento rápido por palavras-chave (sem LLM)
2. Roteamento inteligente via LLM (fallback)
"""

from src.core.config import AGENTES, COORDENADOR_MODELO, COORDENADOR_SYSTEM
from src.core.llm import chamar_coordenador


def rotear_por_palavras_chave(texto: str) -> str | None:
    """
    Tenta rotear usando palavras-chave (sem LLM).
    Retorna None se não tiver confiança suficiente.
    """
    texto_lower = texto.lower()
    pontuacao: dict[str, int] = {}

    for nome, agente in AGENTES.items():
        score = 0
        for palavra in agente["palavras_chave"]:
            if palavra in texto_lower:
                score += 1
        if score > 0:
            pontuacao[nome] = score

    if not pontuacao:
        return None

    melhor = max(pontuacao, key=pontuacao.get)
    scores_sorted = sorted(pontuacao.values(), reverse=True)

    # Empate → deixar pro LLM
    if len(scores_sorted) > 1 and scores_sorted[0] == scores_sorted[1]:
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
