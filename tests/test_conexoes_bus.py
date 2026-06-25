import asyncio

from src.conexoes.bus import InboundMessage, MessageBus, OutboundMessage, SenderInfo


def test_inbound_roundtrip():
    async def cenario():
        bus = MessageBus()
        msg = InboundMessage(
            texto="oi", sender=SenderInfo(id="u1", nome="Nik"),
            canal="cli", chat_id="c1",
        )
        await bus.publicar_entrada(msg)
        recebida = await bus.proxima_entrada()
        return recebida

    recebida = asyncio.run(cenario())
    assert recebida.texto == "oi"
    assert recebida.sender.id == "u1"
    assert recebida.canal == "cli"


def test_saida_notifica_assinantes():
    async def cenario():
        bus = MessageBus()
        recebidas = []

        async def consumidor(m: OutboundMessage):
            recebidas.append(m)

        bus.assinar_saida(consumidor)
        await bus.publicar_saida(OutboundMessage(texto="pong", canal="cli", chat_id="c1"))
        return recebidas

    recebidas = asyncio.run(cenario())
    assert len(recebidas) == 1
    assert recebidas[0].texto == "pong"
