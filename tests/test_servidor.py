from src.conexoes import servidor
from src.core import config


def test_canais_configurados_vazio(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert config.canais_configurados() == []


def test_canais_configurados_telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("TELEGRAM_ALLOW_LIST", "111, 222")
    cfgs = config.canais_configurados()
    assert len(cfgs) == 1
    assert cfgs[0]["tipo"] == "telegram"
    assert cfgs[0]["token"] == "tok123"
    assert cfgs[0]["allow_list"] == ["111", "222"]


def test_montar_servidor_sem_canais(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    _, _, canais = servidor.montar_servidor(processar=lambda a, t: t)
    assert canais == []


def test_montar_servidor_com_telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
    _, _, canais = servidor.montar_servidor(processar=lambda a, t: t)
    assert len(canais) == 1
    assert canais[0].nome == "telegram"
