import asyncio
from src.conexoes.bus import MessageBus, OutboundMessage
from src.conexoes.channels.cli import CLIChannel


def test_receber_publica_inbound():
    async def cenario():
        bus = MessageBus()
        canal = CLIChannel(bus, nome="cli")
        await canal.receber("ola mundo")
        msg = await bus.proxima_entrada()
        return msg

    msg = asyncio.run(cenario())
    assert msg.texto == "ola mundo"
    assert msg.canal == "cli"


def test_send_registra_saida():
    async def cenario():
        bus = MessageBus()
        canal = CLIChannel(bus, nome="cli")
        await canal.send(OutboundMessage(texto="resposta", canal="cli", chat_id="local"))
        return canal.saidas

    saidas = asyncio.run(cenario())
    assert saidas == ["resposta"]
