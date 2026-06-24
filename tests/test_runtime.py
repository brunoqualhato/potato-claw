import asyncio

from src.conexoes.bus import InboundMessage, MessageBus, OutboundMessage, SenderInfo
from src.conexoes.runtime import ERRO_PROCESSAMENTO, Runtime


def test_processar_uma_gera_outbound():
    async def cenario():
        bus = MessageBus()
        recebidas = []

        async def coletor(m: OutboundMessage):
            recebidas.append(m)

        bus.assinar_saida(coletor)

        def processar(agente, pergunta):
            return f"eco: {pergunta}"

        rt = Runtime(bus, processar)
        msg = InboundMessage(
            texto="oi", sender=SenderInfo(id="u1"), canal="cli", chat_id="c1"
        )
        out = await rt.processar_uma(msg)
        return out, recebidas

    out, recebidas = asyncio.run(cenario())
    assert out.texto == "eco: oi"
    assert out.canal == "cli"
    assert out.chat_id == "c1"
    assert len(recebidas) == 1
    assert recebidas[0].texto == "eco: oi"


def test_processar_uma_responde_erro_sem_propagar():
    async def cenario():
        bus = MessageBus()

        def falhar(_agente, _pergunta):
            raise RuntimeError("boom")

        rt = Runtime(bus, falhar)
        msg = InboundMessage(
            texto="oi",
            sender=SenderInfo(id="u1"),
            canal="cli",
            chat_id="c1",
        )
        return await rt.processar_uma(msg)

    out = asyncio.run(cenario())
    assert out.texto == ERRO_PROCESSAMENTO
