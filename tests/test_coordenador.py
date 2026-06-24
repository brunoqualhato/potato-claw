"""Testes para o coordenador de roteamento por palavras-chave."""

import pytest

from src.agentes.coordenador import (
    _baixo_sinal,
    rotear_por_palavras_chave,
    validar_prompt,
)


class TestBaixoSinal:
    """Detecta entradas sem sinal semântico."""

    @pytest.mark.parametrize("entrada", [
        "", "oi", "olá", "ok", "sim", "não", "valeu",
    ])
    def test_entradas_genericas(self, entrada):
        assert _baixo_sinal(entrada) is True

    @pytest.mark.parametrize("entrada", [
        "ab",  # <= 2 tokens e <= 14 chars
        "???!!!",  # muitos símbolos
    ])
    def test_entradas_ruidosas(self, entrada):
        assert _baixo_sinal(entrada) is True

    @pytest.mark.parametrize("entrada", [
        "como criar uma API em python",
        "pesquise sobre inteligência artificial",
        "analise os dados de vendas do trimestre",
    ])
    def test_entradas_com_sinal(self, entrada):
        assert _baixo_sinal(entrada) is False


class TestRotearPorPalavrasChave:
    """Roteamento sem LLM — baseado em keywords."""

    @pytest.mark.parametrize("entrada,esperado", [
        ("crie uma função python para parsear JSON", "programador"),
        ("como fazer deploy na AWS com docker", "programador"),
        ("debug esse erro de typescript", "programador"),
        ("refatorar a classe de autenticação", "programador"),
    ])
    def test_roteamento_programador(self, entrada, esperado):
        assert rotear_por_palavras_chave(entrada) == esperado

    @pytest.mark.parametrize("entrada,esperado", [
        ("pesquise novidades e tendências em IA e machine learning", "pesquisador"),
        ("busque a documentação do FastAPI", "pesquisador"),
        ("qual a temperatura em São Paulo", "pesquisador"),
        ("cotação do dólar hoje", "pesquisador"),
    ])
    def test_roteamento_pesquisador(self, entrada, esperado):
        assert rotear_por_palavras_chave(entrada) == esperado

    @pytest.mark.parametrize("entrada,esperado", [
        ("analise os dados de crescimento do SaaS", "analista"),
        ("qual a estratégia de negócio para VoIP", "analista"),
        ("crie um relatório de métricas", "analista"),
    ])
    def test_roteamento_analista(self, entrada, esperado):
        assert rotear_por_palavras_chave(entrada) == esperado

    def test_sem_confianca_retorna_none(self):
        # Pergunta genérica sem keywords fortes
        resultado = rotear_por_palavras_chave("me ajuda com algo")
        assert resultado is None


class TestValidarPrompt:
    def test_saudacao_e_valida(self):
        valido, _ = validar_prompt("oi")
        assert valido is True

    def test_vazio(self):
        valido, _ = validar_prompt("")
        assert valido is False

    def test_muito_curto(self):
        valido, _ = validar_prompt("ab")
        assert valido is False

    def test_valido(self):
        valido, _ = validar_prompt("como implementar autenticação JWT em python")
        assert valido is True
