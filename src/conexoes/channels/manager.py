"""Gerencia canais e entrega de saída com retry/backoff e split."""
from __future__ import annotations

import asyncio

from src.conexoes.bus import MessageBus, OutboundMessage
from src.conexoes.channels.base import BaseChannel


class ChannelManager:
    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self._canais: dict[str, BaseChannel] = {}
        bus.assinar_saida(self.entregar)

    def adicionar(self, canal: BaseChannel) -> None:
        self._canais[canal.nome] = canal

    async def iniciar_todos(self) -> None:
        for canal in self._canais.values():
            await canal.start()

    async def parar_todos(self) -> None:
        for canal in self._canais.values():
            await canal.stop()

    def _split(self, texto: str, limite: int) -> list[str]:
        if limite <= 0 or len(texto) <= limite:
            return [texto]
        return [texto[i:i + limite] for i in range(0, len(texto), limite)]

    async def entregar(
        self,
        msg: OutboundMessage,
        *,
        max_retries: int = 3,
        base_delay: float = 0.01,
    ) -> bool:
        canal = self._canais.get(msg.canal)
        if canal is None:
            return False

        partes = self._split(msg.texto, canal.max_message_length)
        for parte in partes:
            sub = OutboundMessage(
                texto=parte, canal=msg.canal, chat_id=msg.chat_id, metadata=msg.metadata
            )
            entregue = False
            for tentativa in range(max_retries):
                try:
                    await canal.send(sub)
                    entregue = True
                    break
                except Exception:
                    if tentativa < max_retries - 1:
                        await asyncio.sleep(base_delay * (2 ** tentativa))
            if not entregue:
                return False
        return True
