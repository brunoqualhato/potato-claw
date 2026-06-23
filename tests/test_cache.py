"""Testes para o sistema de cache."""

import tempfile

import pytest

from src.memoria.cache import Cache


@pytest.fixture
def cache_temp():
    """Cache usando arquivo temporário."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        c = Cache(arquivo=f.name)
        yield c


class TestCache:
    def test_salvar_e_buscar(self, cache_temp):
        cache_temp.salvar("qual a capital do Brasil", "Brasília", "generalista")
        resultado = cache_temp.buscar("qual a capital do Brasil")
        assert resultado == "Brasília"

    def test_buscar_inexistente(self, cache_temp):
        assert cache_temp.buscar("pergunta inexistente") is None

    def test_nao_cacheia_saudacoes(self, cache_temp):
        cache_temp.salvar("oi", "Olá!", "generalista")
        assert cache_temp.buscar("oi") is None

    def test_nao_cacheia_genericas(self, cache_temp):
        cache_temp.salvar("ok", "entendi", "generalista")
        assert cache_temp.buscar("ok") is None

    def test_nao_cacheia_consultas_temporais(self, cache_temp):
        pergunta = "pesquisador:qual a cotação do dólar hoje"
        cache_temp.salvar(pergunta, "R$ 5,00", "pesquisador")
        assert cache_temp.buscar(pergunta) is None

    def test_nao_cacheia_consulta_de_versao_sem_palavra_atual(self, cache_temp):
        pergunta = "pesquisador:qual a versão do node"
        cache_temp.salvar(pergunta, "22", "pesquisador")
        assert cache_temp.buscar(pergunta) is None

    def test_limpar(self, cache_temp):
        cache_temp.salvar("teste completo", "resposta", "programador")
        cache_temp.limpar()
        assert cache_temp.buscar("teste completo") is None

    def test_estatisticas(self, cache_temp):
        cache_temp.salvar("pergunta sobre python", "resposta python", "programador")
        stats = cache_temp.estatisticas()
        assert stats["entradas"] == 1
        assert stats["hits_total"] == 1

    def test_hit_incrementa_contador(self, cache_temp):
        cache_temp.salvar("como usar docker compose", "use docker-compose up", "programador")
        cache_temp.buscar("como usar docker compose")
        cache_temp.buscar("como usar docker compose")
        stats = cache_temp.estatisticas()
        assert stats["hits_total"] == 3  # 1 do salvar + 2 buscas

    def test_hash_case_insensitive(self, cache_temp):
        cache_temp.salvar("Como Usar Python", "resposta", "programador")
        resultado = cache_temp.buscar("como usar python")
        assert resultado == "resposta"
