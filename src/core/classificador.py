"""
Classificador de Complexidade.
Decide qual nível de performance usar para cada pergunta.

Nível 1 (Turbo):   Ferramentas + Cache + ChromaDB sem LLM
Nível 2 (Rápido):  Modelo 1.7B com contexto curto
Nível 3 (Profundo): Modelo 4B com RAG completo
"""

import re

from src.core.config import INDICADORES_SIMPLES, INDICADORES_COMPLEXOS


PADRAO_EXPRESSAO = re.compile(r'^[\d\s\+\-\*\/\.\(\)\%\^]+$')
PADRAO_CODIGO = re.compile(r'```|\bdef\b|\bclass\b|\bimport\b|\bSELECT\b|\bFROM\b', re.IGNORECASE)


def classificar_complexidade(texto: str) -> int:
    """
    Classifica a complexidade da pergunta em nível 1, 2 ou 3.
    """
    texto_lower = texto.lower().strip()
    palavras = texto_lower.split()
    num_palavras = len(palavras)

    # ─── NÍVEL 1: Ferramentas diretas ───
    if PADRAO_EXPRESSAO.match(texto.strip()):
        return 1

    palavras_data = ["que horas", "hora atual", "que dia", "data de hoje", "horário"]
    if any(p in texto_lower for p in palavras_data):
        return 1

    if any(p in texto_lower for p in ["quanto é", "calcule", "resultado de", "compute"]):
        return 1

    # ─── SCORE DE COMPLEXIDADE ───
    score_complexo = 0

    score_complexo += sum(2 for ind in INDICADORES_COMPLEXOS if ind in texto_lower)
    score_complexo -= sum(1 for ind in INDICADORES_SIMPLES if ind in texto_lower)

    if PADRAO_CODIGO.search(texto):
        score_complexo += 3

    if num_palavras > 18:
        score_complexo += 1
    if num_palavras > 30:
        score_complexo += 2

    if texto.count("?") > 1:
        score_complexo += 2

    if any(p in texto_lower for p in ["passo a passo", "arquitetura", "trade-off", "benchmark"]):
        score_complexo += 2

    # ─── NÍVEL 3: Complexo ───
    if score_complexo >= 4:
        return 3

    if any(p in texto_lower for p in ["implemente", "crie um", "desenvolva", "código completo", "refatore"]):
        return 3

    # ─── NÍVEL 2: Simples/Rápido ───
    score_simples = sum(1 for ind in INDICADORES_SIMPLES if ind in texto_lower)
    if score_simples >= 1:
        return 2

    if num_palavras <= 10:
        return 2

    if num_palavras <= 20 and score_complexo <= 1:
        return 2

    return 3


def explicar_nivel(nivel: int) -> str:
    """Retorna explicação curta do nível."""
    explicacoes = {
        1: "⚡ Turbo (sem LLM)",
        2: "🚀 Rápido (1.7B)",
        3: "🧠 Profundo (4B + RAG)",
    }
    return explicacoes.get(nivel, "❓ Desconhecido")
