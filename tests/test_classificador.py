"""Testes para o classificador de complexidade."""

import pytest

from src.core.classificador import classificar_complexidade, explicar_nivel


class TestClassificarComplexidade:
    """Testes unitários para classificação de nível."""

    # ─── Nível 1: ferramentas diretas ───

    @pytest.mark.parametrize("entrada", [
        "2 + 2",
        "100 * 3.14",
        "5 + 5 * 2",
        "(10 + 5) / 3",
    ])
    def test_expressoes_matematicas_nivel1(self, entrada):
        assert classificar_complexidade(entrada) == 1

    @pytest.mark.parametrize("entrada", [
        "que horas são",
        "que dia é hoje",
        "hora atual",
        "data de hoje",
    ])
    def test_data_hora_nivel1(self, entrada):
        assert classificar_complexidade(entrada) == 1

    @pytest.mark.parametrize("entrada", [
        "quanto é 50 * 2",
        "calcule 100 / 4",
        "resultado de 7 + 8",
    ])
    def test_calculos_verbais_nivel1(self, entrada):
        assert classificar_complexidade(entrada) == 1

    # ─── Nível 2: perguntas simples ───

    @pytest.mark.parametrize("entrada", [
        "o que é python",
        "defina recursão",
        "explique brevemente o que é REST",
        "liste 3 frameworks web",
    ])
    def test_perguntas_simples_nivel2(self, entrada):
        assert classificar_complexidade(entrada) == 2

    def test_pergunta_curta_nivel2(self):
        assert classificar_complexidade("como usar docker") == 2

    # ─── Nível 3: tarefas complexas ───

    @pytest.mark.parametrize("entrada", [
        "implemente uma API REST com autenticação JWT em Python",
        "crie um sistema de cache distribuído com invalidação",
        "refatore essa classe usando padrões SOLID",
        "código completo de um websocket server",
    ])
    def test_tarefas_complexas_nivel3(self, entrada):
        assert classificar_complexidade(entrada) == 3

    def test_pergunta_com_codigo_nivel3(self):
        # Código inline com contexto longo promove para nível 3
        entrada = "implemente e corrija o bug neste código: ```def foo(): pass``` e explique detalhadamente"
        assert classificar_complexidade(entrada) == 3

    def test_multiplas_perguntas_nivel3(self):
        # Múltiplas perguntas longas com complexidade suficiente
        entrada = "implemente um sistema que responda: qual framework usar? quais as vantagens? como configurar?"
        assert classificar_complexidade(entrada) == 3


class TestExplicarNivel:
    def test_niveis_validos(self):
        assert "Turbo" in explicar_nivel(1)
        assert "Rápido" in explicar_nivel(2)
        assert "Profundo" in explicar_nivel(3)

    def test_nivel_invalido(self):
        assert "Desconhecido" in explicar_nivel(99)
