"""Liga o bus ao motor de agentes: inbound -> resposta -> outbound."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from src.conexoes.bus import InboundMessage, MessageBus, OutboundMessage

Processador = Callable[[str, str], str]
logger = logging.getLogger(__name__)
ERRO_PROCESSAMENTO = "Ops, tive um problema ao processar isso. Pode tentar de novo?"


def _processador_padrao() -> Processador:
    # Import tardio: evita carregar o motor (e Ollama) quando não há canais.
    from src.agentes.executor import SistemaAgentes

    sistema = SistemaAgentes()

    def processar(nome_agente: str, pergunta: str) -> str:
        return sistema.executar(nome_agente, pergunta)

    return processar


class Runtime:
    def __init__(self, bus: MessageBus, processar: Processador | None = None) -> None:
        self._bus = bus
        self._processar = processar or _processador_padrao()
        self.agente_padrao = "generalista"

    async def processar_uma(self, msg: InboundMessage) -> OutboundMessage:
        try:
            resposta = await asyncio.to_thread(self._processar, self.agente_padrao, msg.texto)
        except Exception:
            logger.exception(
                "Erro ao processar mensagem (canal=%s chat=%s)",
                msg.canal,
                msg.chat_id,
            )
            resposta = ERRO_PROCESSAMENTO
        out = OutboundMessage(texto=resposta, canal=msg.canal, chat_id=msg.chat_id)
        await self._bus.publicar_saida(out)
        return out

    async def rodar(self) -> None:
        while True:
            msg = await self._bus.proxima_entrada()
            try:
                await self.processar_uma(msg)
            except Exception:
                logger.exception("Erro no loop do runtime; mensagem ignorada")
