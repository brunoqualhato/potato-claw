"""Canal Telegram via Bot API (long-polling), sem dependência externa (urllib)."""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from src.conexoes.bus import InboundMessage, MessageBus, OutboundMessage, SenderInfo
from src.conexoes.channels.base import BaseChannel
from src.conexoes.channels.registry import registrar

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/{metodo}"
# Erros em que não adianta re-tentar: token inválido / bot bloqueado.
_ERROS_FATAIS = {401, 403, 404}
# Espera entre tentativas após erro transitório (rate limit, rede, 5xx).
_BACKOFF_ERRO_S = 5


class TelegramChannel(BaseChannel):
    def __init__(
        self,
        bus: MessageBus,
        *,
        token: str,
        nome: str = "telegram",
        allow_list=None,
        max_message_length: int = 4096,
    ) -> None:
        super().__init__(bus, nome=nome, allow_list=allow_list,
                         max_message_length=max_message_length)
        self._token = token
        self._rodando = False
        self._offset = 0

    def _api_call(self, metodo: str, params: dict) -> dict:
        """Chama a Bot API. Sempre retorna um dict (o Telegram responde erros
        com corpo JSON util, ex.: {"ok": false, "error_code": 401, ...}).
        Só erros de rede/timeout (URLError) propagam como excecao."""
        url = _API.format(token=self._token, metodo=metodo)
        data = urllib.parse.urlencode(params).encode()
        try:
            with urllib.request.urlopen(url, data=data, timeout=65) as resp:  # noqa: S310
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            # urlopen levanta em 4xx/5xx, mas o corpo tem o JSON de erro do Telegram.
            try:
                return json.loads(e.read().decode())
            except Exception:
                return {"ok": False, "error_code": e.code, "description": str(e)}

    def _inbound_de_update(self, update: dict) -> InboundMessage | None:
        msg = update.get("message") or {}
        texto = msg.get("text")
        if not texto:
            return None
        remetente = msg.get("from", {})
        sender = SenderInfo(
            id=str(remetente.get("id", "")),
            nome=remetente.get("first_name", ""),
            canal=self.nome,
        )
        if not self.is_allowed_sender(sender):
            return None
        return InboundMessage(
            texto=texto,
            sender=sender,
            canal=self.nome,
            chat_id=str(msg.get("chat", {}).get("id", "")),
        )

    async def send(self, msg: OutboundMessage) -> None:
        resp = await asyncio.to_thread(
            self._api_call, "sendMessage", {"chat_id": msg.chat_id, "text": msg.texto}
        )
        if not resp.get("ok", False):
            # Levanta para o ChannelManager aplicar retry/backoff.
            raise RuntimeError(
                f"Telegram sendMessage falhou: {resp.get('description', resp)}"
            )

    async def start(self) -> None:
        self._rodando = True
        while self._rodando:
            try:
                resp = await asyncio.to_thread(
                    self._api_call, "getUpdates", {"offset": self._offset, "timeout": 60}
                )
            except Exception as e:  # erro de rede/timeout: não derruba o canal
                logger.warning("Telegram getUpdates erro de rede: %s", e)
                await asyncio.sleep(_BACKOFF_ERRO_S)
                continue

            if not resp.get("ok", False):
                code = resp.get("error_code")
                logger.warning(
                    "Telegram getUpdates falhou (%s): %s", code, resp.get("description")
                )
                if code in _ERROS_FATAIS:
                    logger.error("Erro fatal do Telegram (%s). Parando o canal.", code)
                    self._rodando = False
                    break
                await asyncio.sleep(_BACKOFF_ERRO_S)
                continue

            for update in resp.get("result", []):
                self._offset = update["update_id"] + 1
                inbound = self._inbound_de_update(update)
                if inbound is not None:
                    await self._bus.publicar_entrada(inbound)

    async def stop(self) -> None:
        self._rodando = False


registrar("telegram", lambda bus, cfg: TelegramChannel(
    bus, token=cfg["token"], allow_list=cfg.get("allow_list")
))
