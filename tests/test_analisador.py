"""Testes para o analisador de intenção (parsing de JSON, sem Ollama)."""

from unittest.mock import patch

from src.core.analisador import (
    IntencaoAnalisada,
    _corrigir_inconsistencias,
    _parse_resposta,
    analisar_intencao,
)
from src.core.config import KEEP_ALIVE_PRINCIPAL


class TestParseResposta:
    """Testa o parser de JSON do analisador."""

    def test_json_valido_completo(self):
        raw = '{"agente":"programador","precisa_web":false,"ferramenta":null,"parametros":{}}'
        r = _parse_resposta(raw)
        assert r.agente == "programador"
        assert r.precisa_web is False
        assert r.ferramenta is None
        assert r.parametros == {}

    def test_json_com_ferramenta_e_params(self):
        raw = '{"agente":"generalista","precisa_web":false,"ferramenta":"data_hora","parametros":{"local":"tokyo"}}'
        r = _parse_resposta(raw)
        assert r.ferramenta == "data_hora"
        assert r.parametros["local"] == "tokyo"

    def test_json_com_web_true(self):
        raw = '{"agente":"pesquisador","precisa_web":true,"ferramenta":null,"parametros":{}}'
        r = _parse_resposta(raw)
        assert r.precisa_web is True
        assert r.agente == "pesquisador"

    def test_json_com_lixo_ao_redor(self):
        raw = 'Aqui vai: {"agente":"analista","precisa_web":false,"ferramenta":null,"parametros":{}} fim'
        r = _parse_resposta(raw)
        assert r.agente == "analista"

    def test_json_aninhado(self):
        raw = '{"agente":"generalista","precisa_web":false,"ferramenta":"calculo","parametros":{"expressao":"2+2"}}'
        r = _parse_resposta(raw)
        assert r.ferramenta == "calculo"
        assert r.parametros["expressao"] == "2+2"

    def test_sem_json(self):
        r = _parse_resposta("texto sem json nenhum")
        assert r.agente == "generalista"
        assert r.precisa_web is False

    def test_json_invalido(self):
        r = _parse_resposta("{agente: programador}")  # JSON inválido
        assert r.agente == "generalista"

    def test_agente_invalido_normaliza(self):
        raw = '{"agente":"hacker","precisa_web":false,"ferramenta":null,"parametros":{}}'
        r = _parse_resposta(raw)
        assert r.agente == "generalista"  # Normaliza para generalista

    def test_ferramenta_invalida_normaliza(self):
        raw = '{"agente":"generalista","precisa_web":false,"ferramenta":"exploit","parametros":{}}'
        r = _parse_resposta(raw)
        assert r.ferramenta is None  # Normaliza para None

    def test_parametros_nao_dict(self):
        raw = '{"agente":"generalista","precisa_web":false,"ferramenta":null,"parametros":"invalido"}'
        r = _parse_resposta(raw)
        assert r.parametros == {}


@patch("src.core.analisador._obter_exemplos_dinamicos", return_value="")
@patch("src.core.analisador.chamar_llm")
def test_analisador_mantem_modelo_rapido_residente(mock_llm, _mock_exemplos):
    mock_llm.return_value = {
        "resposta": '{"agente":"generalista","precisa_web":false,"ferramenta":null,"parametros":{}}',
        "erro": False,
    }

    analisar_intencao("explique cache")

    kwargs = mock_llm.call_args.kwargs
    assert kwargs["keep_alive"] == KEEP_ALIVE_PRINCIPAL
    assert kwargs["formato"] == "json"


def test_guardrail_corrige_receita_e_forca_grounding_web():
    intencao = IntencaoAnalisada(agente="programador", precisa_web=True)

    corrigida = _corrigir_inconsistencias("como fazer um pão de queijo", intencao)

    assert corrigida.agente == "generalista"
    assert corrigida.precisa_web is True
    assert "ingredientes" in corrigida.parametros["consulta_web"]
    assert corrigida.parametros["resposta_web_direta"] is True


def test_guardrail_preserva_programacao_quando_ha_sinal_tecnico():
    intencao = IntencaoAnalisada(agente="programador", precisa_web=False)

    corrigida = _corrigir_inconsistencias("como fazer deploy de uma API", intencao)

    assert corrigida.agente == "programador"


def test_guardrail_preserva_web_em_consulta_temporal():
    intencao = IntencaoAnalisada(agente="pesquisador", precisa_web=True)

    corrigida = _corrigir_inconsistencias("qual a versão atual do Python", intencao)

    assert corrigida.precisa_web is True


@patch("src.core.analisador._obter_exemplos_dinamicos", return_value="")
@patch("src.core.analisador.chamar_llm")
def test_analisador_aplica_guardrail_ao_resultado_do_modelo(mock_llm, _mock_exemplos):
    mock_llm.return_value = {
        "resposta": '{"agente":"programador","precisa_web":true,"ferramenta":null,"parametros":{}}',
        "erro": False,
    }

    resultado = analisar_intencao("como fazer um pão de queijo")

    assert resultado.agente == "generalista"
    assert resultado.precisa_web is True
