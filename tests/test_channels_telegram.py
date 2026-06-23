import asyncio
import io
import urllib.error
from unittest.mock import patch

import pytest

from src.conexoes.bus import MessageBus, OutboundMessage
from src.conexoes.channels.telegram import TelegramChannel


def test_parse_update_vira_inbound():
    bus = MessageBus()
    canal = TelegramChannel(bus, token="t")
    update = {
        "message": {
            "text": "oi bot",
            "chat": {"id": 123},
            "from": {"id": 99, "first_name": "Nik"},
        }
    }
    msg = canal._inbound_de_update(update)
    assert msg is not None
    assert msg.texto == "oi bot"
    assert msg.chat_id == "123"
    assert msg.sender.id == "99"
    assert msg.canal == "telegram"


def test_allow_list_bloqueia_update():
    bus = MessageBus()
    canal = TelegramChannel(bus, token="t", allow_list=["1"])
    update = {"message": {"text": "x", "chat": {"id": 5}, "from": {"id": 99}}}
    assert canal._inbound_de_update(update) is None


def test_send_chama_sendmessage():
    async def cenario():
        bus = MessageBus()
        canal = TelegramChannel(bus, token="t")
        with patch.object(canal, "_api_call", return_value={"ok": True}) as mock_api:
            await canal.send(OutboundMessage(texto="oi", canal="telegram", chat_id="123"))
        return mock_api

    mock_api = asyncio.run(cenario())
    metodo, params = mock_api.call_args[0]
    assert metodo == "sendMessage"
    assert params["chat_id"] == "123"
    assert params["text"] == "oi"


def test_api_call_trata_http_error_retornando_json():
    """HTTPError (401/4xx) nao deve estourar: retorna o JSON de erro do Telegram."""
    corpo = b'{"ok":false,"error_code":401,"description":"Unauthorized"}'
    err = urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, io.BytesIO(corpo))
    canal = TelegramChannel(MessageBus(), token="t")
    with patch("urllib.request.urlopen", side_effect=err):
        resp = canal._api_call("getMe", {})
    assert resp["ok"] is False
    assert resp["error_code"] == 401


def test_send_levanta_quando_ok_false():
    """send deve levantar em erro logico para o ChannelManager re-tentar."""
    async def cenario():
        canal = TelegramChannel(MessageBus(), token="t")
        with patch.object(canal, "_api_call", return_value={"ok": False, "description": "chat not found"}):
            await canal.send(OutboundMessage(texto="oi", canal="telegram", chat_id="999"))

    with pytest.raises(RuntimeError):
        asyncio.run(cenario())


def test_start_para_em_erro_fatal_sem_loop():
    """getUpdates com 401 (fatal) para o canal em vez de entrar em loop infinito."""
    async def cenario():
        canal = TelegramChannel(MessageBus(), token="t")
        with patch.object(
            canal, "_api_call",
            return_value={"ok": False, "error_code": 401, "description": "Unauthorized"},
        ):
            await asyncio.wait_for(canal.start(), timeout=2)
        return canal._rodando

    assert asyncio.run(cenario()) is False
