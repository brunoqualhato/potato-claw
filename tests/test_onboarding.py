from src.onboarding import resumo_setup


def test_resumo_setup_lista_perfil_modelos_canais():
    texto = resumo_setup(
        perfil_ativo="ultra_leve",
        modelos={"rapido": "lfm2.5", "completo": "qwen3:4b"},
        canais_disponiveis=["cli", "telegram"],
    )
    assert "ultra_leve" in texto
    assert "lfm2.5" in texto
    assert "telegram" in texto
    assert "cli" in texto
