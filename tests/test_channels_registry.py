import pytest
from src.conexoes.bus import MessageBus, OutboundMessage
from src.conexoes.channels.base import BaseChannel
from src.conexoes.channels import registry


class _Falso(BaseChannel):
    async def start(self): ...
    async def stop(self): ...
    async def send(self, msg: OutboundMessage): ...


def test_registrar_e_criar():
    registry.registrar("falso", lambda bus, cfg: _Falso(bus, nome="falso"))
    c = registry.criar("falso", MessageBus(), {})
    assert isinstance(c, _Falso)
    assert "falso" in registry.disponiveis()


def test_criar_invalido_levanta():
    with pytest.raises(ValueError):
        registry.criar("inexistente", MessageBus(), {})
