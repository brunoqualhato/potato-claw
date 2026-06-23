"""Testes da deteccao de RAM e do perfil sugerido (tuning hardware fraco)."""
import src.core.config as config


def test_detectar_ram_gb_retorna_float_nao_negativo():
    ram = config._detectar_ram_gb()
    assert isinstance(ram, float)
    assert ram >= 0.0


def test_num_ctx_nivel_definido_para_2_e_3():
    assert config.NUM_CTX_NIVEL[2] > 0
    assert config.NUM_CTX_NIVEL[3] > 0
    assert config.NUM_CTX_NIVEL[3] >= config.NUM_CTX_NIVEL[2]


def test_perfil_sugerido_ultra_leve(monkeypatch):
    monkeypatch.setattr(config, "RAM_GB", 8.0)
    assert config.perfil_sugerido_por_ram() == "ultra_leve"


def test_perfil_sugerido_equilibrado(monkeypatch):
    monkeypatch.setattr(config, "RAM_GB", 12.0)
    assert config.perfil_sugerido_por_ram() == "equilibrado"


def test_perfil_sugerido_maximo(monkeypatch):
    monkeypatch.setattr(config, "RAM_GB", 32.0)
    assert config.perfil_sugerido_por_ram() == "maximo"


def test_perfil_sugerido_none_quando_ram_desconhecida(monkeypatch):
    monkeypatch.setattr(config, "RAM_GB", 0.0)
    assert config.perfil_sugerido_por_ram() is None
