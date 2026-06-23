"""Liga o bus ao motor de agentes: inbound -> resposta -> outbound."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from src.conexoes.bus import InboundMessage, MessageBus, OutboundMessage

logger = logging.getLogger(__name__)

Processador = Callable[[str, str, str], str]

# Resposta enviada ao usuario quando o processamento falha (em vez de cair calado).
ERRO_PROCESSAMENTO = "Ops, tive um problema ao processar isso. Pode tentar de novo?"


def _processador_padrao() -> Processador:
    # Import tardio: evita carregar o motor (e Ollama) quando não há canais.
    from src.agentes.executor import SistemaAgentes

    sistema = SistemaAgentes()

    def processar(nome_agente: str, pergunta: str, sessao: str = "") -> str:
        # Isola o historico por sessao (canal:pessoa) em multi-user. "" = global.
        sistema.memoria.sessao_ativa = sessao
        return sistema.executar(nome_agente, pergunta)

    return processar


class Runtime:
    def __init__(
        self,
        bus: MessageBus,
        processar: Processador | None = None,
        manager=None,
    ) -> None:
        self._bus = bus
        self._processar = processar or _processador_padrao()
        self._manager = manager
        self.agente_padrao = "generalista"
        self.typing_intervalo_s = 4.0

    async def _loop_typing(self, canal: str, chat_id: str) -> None:
        """Mantem 'digitando...' ativo enquanto o LLM processa (reenvia periodicamente)."""
        try:
            while True:
                await self._manager.sinalizar_typing(canal, chat_id)
                await asyncio.sleep(self.typing_intervalo_s)
        except asyncio.CancelledError:
            pass

    async def processar_uma(self, msg: InboundMessage) -> OutboundMessage:
        typing_task = None
        if self._manager is not None:
            typing_task = asyncio.create_task(self._loop_typing(msg.canal, msg.chat_id))
        sessao = f"{msg.canal}:{msg.sender.id}"
        try:
            resposta = await asyncio.to_thread(
                self._processar, self.agente_padrao, msg.texto, sessao
            )
        except Exception:
            # Uma mensagem que falha NUNCA pode derrubar o agente: loga e responde erro.
            logger.exception("Erro ao processar mensagem (canal=%s chat=%s)", msg.canal, msg.chat_id)
            resposta = ERRO_PROCESSAMENTO
        finally:
            if typing_task is not None:
                typing_task.cancel()
        out = OutboundMessage(texto=resposta, canal=msg.canal, chat_id=msg.chat_id)
        await self._bus.publicar_saida(out)
        return out

    async def rodar(self) -> None:
        while True:
            msg = await self._bus.proxima_entrada()
            try:
                await self.processar_uma(msg)
            except Exception:
                # Defesa em profundidade: o loop do servidor nunca para por uma mensagem.
                logger.exception("Erro no loop do runtime (mensagem ignorada)")
