import asyncio
import time

from src.conexoes.bus import InboundMessage, MessageBus, OutboundMessage, SenderInfo
from src.conexoes.runtime import Runtime


def test_processar_uma_gera_outbound():
    async def cenario():
        bus = MessageBus()
        recebidas = []

        async def coletor(m: OutboundMessage):
            recebidas.append(m)

        bus.assinar_saida(coletor)

        def processar(agente, pergunta, sessao=""):
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


def test_runtime_sinaliza_typing_enquanto_processa():
    """Com manager, o Runtime mantem 'digitando...' durante o processamento."""
    async def cenario():
        bus = MessageBus()
        chamadas = []

        class FakeManager:
            async def sinalizar_typing(self, canal, chat_id):
                chamadas.append((canal, chat_id))

        def processar(agente, pergunta, sessao=""):
            time.sleep(0.05)  # da tempo do loop de typing rodar
            return "ok"

        rt = Runtime(bus, processar, manager=FakeManager())
        rt.typing_intervalo_s = 0.01
        await rt.processar_uma(
            InboundMessage(texto="oi", sender=SenderInfo(id="u1"),
                           canal="telegram", chat_id="c1")
        )
        return chamadas

    chamadas = asyncio.run(cenario())
    assert ("telegram", "c1") in chamadas


def test_runtime_nao_morre_quando_processar_falha():
    """Uma mensagem que estoura nao pode derrubar o agente: responde erro e segue."""
    from src.conexoes.runtime import ERRO_PROCESSAMENTO

    async def cenario():
        bus = MessageBus()
        recebidas = []

        async def coletor(m: OutboundMessage):
            recebidas.append(m)

        bus.assinar_saida(coletor)

        def processar(agente, texto, sessao=""):
            raise RuntimeError("boom")

        rt = Runtime(bus, processar)
        out = await rt.processar_uma(
            InboundMessage(texto="x", sender=SenderInfo(id="u"), canal="cli", chat_id="c")
        )
        return out, recebidas

    out, recebidas = asyncio.run(cenario())
    assert out.texto == ERRO_PROCESSAMENTO
    assert len(recebidas) == 1


def test_runtime_passa_sessao_canal_pessoa():
    """O runtime deriva a sessao de canal:pessoa e passa ao processador."""
    async def cenario():
        bus = MessageBus()
        capturado = {}

        def processar(agente, pergunta, sessao=""):
            capturado["sessao"] = sessao
            return "ok"

        rt = Runtime(bus, processar)
        await rt.processar_uma(
            InboundMessage(texto="oi", sender=SenderInfo(id="42"),
                           canal="telegram", chat_id="c1")
        )
        return capturado

    capturado = asyncio.run(cenario())
    assert capturado["sessao"] == "telegram:42"
