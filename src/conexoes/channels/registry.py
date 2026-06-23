"""Registro de factories de canal."""
from __future__ import annotations

from collections.abc import Callable

from src.conexoes.bus import MessageBus
from src.conexoes.channels.base import BaseChannel

ChannelFactory = Callable[[MessageBus, dict], BaseChannel]

_factories: dict[str, ChannelFactory] = {}


def registrar(nome: str, factory: ChannelFactory) -> None:
    _factories[nome] = factory


def criar(nome: str, bus: MessageBus, config: dict) -> BaseChannel:
    if nome not in _factories:
        raise ValueError(f"Canal '{nome}' nao registrado. Disponiveis: {list(_factories)}")
    return _factories[nome](bus, config)


def disponiveis() -> list[str]:
    return list(_factories)
