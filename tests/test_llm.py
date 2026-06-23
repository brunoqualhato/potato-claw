"""Testes das otimizações de carregamento de modelos."""

from unittest.mock import patch

from src.core.llm import warmup_modelos


@patch("src.core.llm.ollama")
def test_warmup_aquece_apenas_funcoes_selecionadas(mock_ollama):
    modelos = {
        "rapido": "modelo-rapido",
        "completo": "modelo-profundo",
        "embedding": "modelo-embedding",
    }

    warmup_modelos(modelos, funcoes=("rapido",))

    mock_ollama.chat.assert_called_once()
    mock_ollama.embeddings.assert_not_called()


@patch("src.core.llm.ollama")
def test_warmup_embedding_usa_api_de_embedding(mock_ollama):
    warmup_modelos({"embedding": "modelo-embedding"}, funcoes=("embedding",))

    mock_ollama.embeddings.assert_called_once_with(
        model="modelo-embedding",
        prompt="warmup",
    )
    mock_ollama.chat.assert_not_called()
