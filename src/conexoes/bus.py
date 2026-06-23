"""Barramento de mensagens assíncrono e leve (asyncio, sem broker externo)."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SenderInfo:
    id: str
    nome: str = ""
    canal: str = ""


@dataclass(frozen=True)
class InboundMessage:
    texto: str
    sender: SenderInfo
    canal: str
    chat_id: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OutboundMessage:
    texto: str
    canal: str
    chat_id: str
    metadata: dict = field(default_factory=dict)


SaidaCallback = Callable[[OutboundMessage], Awaitable[None]]


class MessageBus:
    """Entrada via fila (1 consumidor: o coordenador). Saída via pub/sub (N canais)."""

    def __init__(self) -> None:
        self._entrada: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self._assinantes_saida: list[SaidaCallback] = []

    async def publicar_entrada(self, msg: InboundMessage) -> None:
        await self._entrada.put(msg)

    async def proxima_entrada(self) -> InboundMessage:
        return await self._entrada.get()

    def assinar_saida(self, callback: SaidaCallback) -> None:
        self._assinantes_saida.append(callback)

    async def publicar_saida(self, msg: OutboundMessage) -> None:
        for cb in self._assinantes_saida:
            await cb(msg)
