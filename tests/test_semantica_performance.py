"""Regressões das otimizações de embeddings."""

from unittest.mock import patch

from src.memoria.semantica import MemoriaSemantica


def test_embedding_recente_e_reutilizado():
    MemoriaSemantica.resetar_instancia()
    memoria = MemoriaSemantica()

    with patch("src.memoria.semantica.ollama_client.embeddings") as embeddings:
        embeddings.return_value = {"embedding": [0.1, 0.2]}

        primeiro = memoria._gerar_embedding("mesma pergunta")
        segundo = memoria._gerar_embedding("mesma pergunta")

    assert primeiro == segundo
    embeddings.assert_called_once()
    MemoriaSemantica.resetar_instancia()
