"""Registro nome -> provider. Default = ollama."""
from __future__ import annotations

from src.provedores.base import LLMProvider

_provedores: dict[str, type[LLMProvider]] = {}


def registrar(nome: str, classe: type[LLMProvider]) -> None:
    _provedores[nome] = classe


def criar(nome: str = "ollama", **kwargs) -> LLMProvider:
    if nome not in _provedores:
        raise ValueError(
            f"Provider '{nome}' nao registrado. Disponiveis: {list(_provedores)}"
        )
    return _provedores[nome](**kwargs)


def disponiveis() -> list[str]:
    return list(_provedores)
