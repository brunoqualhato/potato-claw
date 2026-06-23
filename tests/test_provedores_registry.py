import pytest

import src.provedores  # noqa: F401  (dispara auto-registro)
from src.provedores import registry
from src.provedores.ollama_provider import OllamaProvider


def test_ollama_registrado_por_padrao():
    assert "ollama" in registry.disponiveis()


def test_criar_default_retorna_ollama():
    p = registry.criar()
    assert isinstance(p, OllamaProvider)
    assert p.nome == "ollama"


def test_criar_nome_invalido_levanta():
    with pytest.raises(ValueError):
        registry.criar("nao-existe")
