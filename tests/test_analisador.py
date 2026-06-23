"""Testes para o analisador de intenção (parsing de JSON, sem Ollama)."""

from src.core.analisador import _parse_resposta


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
