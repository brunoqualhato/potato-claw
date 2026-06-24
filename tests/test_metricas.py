"""Testes das métricas orientadas a performance."""

from src.memoria.sqlite import Memoria


def test_metricas_incluem_percentis_e_tokens(tmp_path):
    memoria = Memoria(str(tmp_path / "metricas.db"))
    try:
        memoria.salvar_metrica("generalista", 2, 100, 10, 5, "llm_rapido")
        memoria.salvar_metrica("generalista", 2, 300, 20, 15, "llm_rapido")

        resumo = memoria.metricas_resumo()[2]

        assert resumo["avg_ms"] == 200
        assert resumo["p50_ms"] == 200
        assert resumo["p95_ms"] == 300
        assert resumo["tokens_entrada"] == 30
        assert resumo["tokens_saida"] == 20
    finally:
        memoria.fechar()
