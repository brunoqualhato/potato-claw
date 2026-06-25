"""Contrato comum de um canal de mensageria."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.conexoes.bus import MessageBus, OutboundMessage, SenderInfo


class BaseChannel(ABC):
    def __init__(
        self,
        bus: MessageBus,
        *,
        nome: str,
        allow_list: list[str] | None = None,
        max_message_length: int = 0,
    ) -> None:
        self._bus = bus
        self._nome = nome
        self._allow_list = list(allow_list or [])
        self.max_message_length = max_message_length

    @property
    def nome(self) -> str:
        return self._nome

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        ...

    def is_allowed_sender(self, sender: SenderInfo) -> bool:
        if not self._allow_list:
            return True
        return sender.id in self._allow_list
