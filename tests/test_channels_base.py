import pytest

from src.conexoes.bus import MessageBus, OutboundMessage, SenderInfo
from src.conexoes.channels.base import BaseChannel


class CanalDummy(BaseChannel):
    async def start(self):
        self._iniciado = True

    async def stop(self):
        self._iniciado = False

    async def send(self, msg: OutboundMessage):
        self.enviadas = getattr(self, "enviadas", [])
        self.enviadas.append(msg)


def test_base_channel_nao_instancia():
    with pytest.raises(TypeError):
        BaseChannel(MessageBus(), nome="x")


def test_allow_list_vazia_libera_todos():
    c = CanalDummy(MessageBus(), nome="dummy")
    assert c.is_allowed_sender(SenderInfo(id="qualquer")) is True


def test_allow_list_restringe():
    c = CanalDummy(MessageBus(), nome="dummy", allow_list=["u1"])
    assert c.is_allowed_sender(SenderInfo(id="u1")) is True
    assert c.is_allowed_sender(SenderInfo(id="u2")) is False


def test_nome_exposto():
    c = CanalDummy(MessageBus(), nome="dummy")
    assert c.nome == "dummy"
