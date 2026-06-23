import asyncio
from src.conexoes.bus import MessageBus, OutboundMessage
from src.conexoes.channels.base import BaseChannel
from src.conexoes.channels.manager import ChannelManager


class CanalFalho(BaseChannel):
    def __init__(self, bus, *, nome, falhas=0, max_message_length=0):
        super().__init__(bus, nome=nome, max_message_length=max_message_length)
        self.falhas_restantes = falhas
        self.enviadas = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send(self, msg):
        if self.falhas_restantes > 0:
            self.falhas_restantes -= 1
            raise RuntimeError("falha temporaria")
        self.enviadas.append(msg.texto)


def test_entrega_simples():
    async def cenario():
        bus = MessageBus()
        mgr = ChannelManager(bus)
        canal = CanalFalho(bus, nome="cli")
        mgr.adicionar(canal)
        ok = await mgr.entregar(OutboundMessage(texto="oi", canal="cli", chat_id="c1"))
        return ok, canal.enviadas

    ok, enviadas = asyncio.run(cenario())
    assert ok is True
    assert enviadas == ["oi"]


def test_retry_recupera_apos_falhas():
    async def cenario():
        bus = MessageBus()
        mgr = ChannelManager(bus)
        canal = CanalFalho(bus, nome="cli", falhas=2)
        mgr.adicionar(canal)
        ok = await mgr.entregar(
            OutboundMessage(texto="oi", canal="cli", chat_id="c1"),
            max_retries=3, base_delay=0.0,
        )
        return ok, canal.enviadas

    ok, enviadas = asyncio.run(cenario())
    assert ok is True
    assert enviadas == ["oi"]


def test_split_mensagem_longa():
    async def cenario():
        bus = MessageBus()
        mgr = ChannelManager(bus)
        canal = CanalFalho(bus, nome="cli", max_message_length=5)
        mgr.adicionar(canal)
        await mgr.entregar(OutboundMessage(texto="abcdefghij", canal="cli", chat_id="c1"))
        return canal.enviadas

    enviadas = asyncio.run(cenario())
    assert enviadas == ["abcde", "fghij"]


def test_canal_inexistente_retorna_false():
    async def cenario():
        bus = MessageBus()
        mgr = ChannelManager(bus)
        return await mgr.entregar(OutboundMessage(texto="x", canal="nao-existe", chat_id="c"))

    assert asyncio.run(cenario()) is False
