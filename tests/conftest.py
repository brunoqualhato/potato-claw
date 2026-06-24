"""
Fixtures compartilhadas para isolamento de testes.
Garante que testes não toquem a base de dados/cache real.
"""


import pytest


@pytest.fixture(autouse=True)
def isolar_chromadb_singleton():
    """
    Reseta o singleton do MemoriaSemantica entre testes para evitar
    estado compartilhado e acesso ao ChromaDB real.
    """
    from src.memoria.semantica import MemoriaSemantica
    # Salva e reseta
    instancia_anterior = MemoriaSemantica._instancia
    MemoriaSemantica._instancia = None
    yield
    # Restaura
    MemoriaSemantica._instancia = instancia_anterior
