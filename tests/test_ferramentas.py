"""Testes para as ferramentas de execução pré-LLM."""

import pytest

from src.ferramentas.resolver import (
    _RESPOSTAS_SAUDACAO,
    calcular,
    comando_permitido,
    executar_ferramentas,
    verificar_ferramenta_calculo,
    verificar_ferramenta_data,
    verificar_ferramenta_saudacao,
)


class TestComandoPermitido:
    def test_binario_permitido(self):
        assert comando_permitido("ls -la") is True
        assert comando_permitido("echo oi") is True

    def test_pergunta_natural_nao_e_comando(self):
        assert comando_permitido("Qual modelo de IA") is False
        assert comando_permitido("Como você funciona") is False

    def test_vazio_ou_invalido(self):
        assert comando_permitido("") is False
        assert comando_permitido('aspas "abertas') is False


class TestCalcular:
    @pytest.mark.parametrize("expr,esperado", [
        ("2 + 2", 4),
        ("10 * 3", 30),
        ("100 / 4", 25.0),
        ("2 ** 10", 1024),
        ("sqrt(16)", 4.0),
        ("pi", 3.141592653589793),
    ])
    def test_calculos_validos(self, expr, esperado):
        resultado = calcular(expr)
        assert resultado is not None
        assert str(esperado) in resultado

    @pytest.mark.parametrize("expr", [
        "hello world",
        "como vai",
        "import os",
    ])
    def test_nao_matematico_retorna_none(self, expr):
        assert calcular(expr) is None

    def test_expressao_invalida(self):
        # Divisão por zero retorna None (exception)
        resultado = calcular("1/0")
        assert resultado is None


class TestVerificarFerramentaData:
    @pytest.mark.parametrize("entrada", [
        "que horas são",
        "hora atual",
        "que dia é hoje",
        "data de hoje",
    ])
    def test_detecta_pergunta_data(self, entrada):
        resultado = verificar_ferramenta_data(entrada)
        assert resultado is not None
        assert "Data:" in resultado or "Hora:" in resultado

    def test_nao_detecta_outro_texto(self):
        assert verificar_ferramenta_data("como programar em python") is None


class TestVerificarFerramentaCalculo:
    def test_quanto_e(self):
        resultado = verificar_ferramenta_calculo("quanto é 5 + 5")
        assert resultado is not None
        assert "10" in resultado

    def test_calcule(self):
        resultado = verificar_ferramenta_calculo("calcule 100 * 2")
        assert resultado is not None
        assert "200" in resultado

    def test_expressao_pura(self):
        resultado = verificar_ferramenta_calculo("3 + 7")
        assert resultado is not None
        assert "10" in resultado

    def test_texto_normal_retorna_none(self):
        assert verificar_ferramenta_calculo("o que é python") is None


class TestVerificarFerramentaSaudacao:
    @pytest.mark.parametrize("entrada", [
        "oi", "olá", "bom dia", "boa tarde", "hello", "OI", " e aí ",
    ])
    def test_saudacoes_puras_retornam_variante_com_identidade(self, entrada):
        resultado = verificar_ferramenta_saudacao(entrada)
        assert resultado in _RESPOSTAS_SAUDACAO
        assert "potato-claw" in resultado.lower()

    def test_nao_saudacao(self):
        assert verificar_ferramenta_saudacao("crie um script python") is None

    @pytest.mark.parametrize("entrada", [
        "beleza e você?", "como você se chama?", "qual seu nome?", "tudo bem com você?",
    ])
    def test_pergunta_nao_e_saudacao_pura(self, entrada):
        # Perguntas reais NAO devem virar resposta rapida; caem pro LLM.
        assert verificar_ferramenta_saudacao(entrada) is None


class TestExecutarFerramentas:
    def test_data_hora(self):
        resultado = executar_ferramentas("que horas são")
        assert resultado is not None

    def test_calculo(self):
        resultado = executar_ferramentas("quanto é 2 + 2")
        assert resultado is not None
        assert "4" in resultado

    def test_saudacao(self):
        resultado = executar_ferramentas("oi")
        assert resultado is not None

    def test_sem_ferramenta_retorna_none(self):
        resultado = executar_ferramentas("explique o teorema de pitágoras detalhadamente")
        assert resultado is None
