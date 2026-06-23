"""Canal CLI: stdin vira inbound, resposta sai no console."""
from __future__ import annotations

from rich.console import Console

from src.conexoes.bus import InboundMessage, MessageBus, OutboundMessage, SenderInfo
from src.conexoes.channels.base import BaseChannel
from src.conexoes.channels.registry import registrar

console = Console()


class CLIChannel(BaseChannel):
    def __init__(self, bus: MessageBus, *, nome: str = "cli", allow_list=None) -> None:
        super().__init__(bus, nome=nome, allow_list=allow_list)
        self.saidas: list[str] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def receber(self, texto: str) -> None:
        msg = InboundMessage(
            texto=texto,
            sender=SenderInfo(id="local", nome="local", canal=self.nome),
            canal=self.nome,
            chat_id="local",
        )
        await self._bus.publicar_entrada(msg)

    async def send(self, msg: OutboundMessage) -> None:
        self.saidas.append(msg.texto)
        console.print(msg.texto)


registrar("cli", lambda bus, cfg: CLIChannel(bus, allow_list=cfg.get("allow_list")))
