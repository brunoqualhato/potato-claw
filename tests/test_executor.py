"""
Testes de integração para o pipeline principal (executor).
Usa mocks do Ollama para testar o fluxo end-to-end sem GPU.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agentes.executor import SistemaAgentes
from src.memoria.cache import Cache
from src.memoria.sqlite import Memoria


@pytest.fixture
def sistema_mock(tmp_path):
    """Sistema com memória e cache em tempfiles, ChromaDB mockado."""
    db_path = str(tmp_path / "test_memoria.db")
    cache_path = str(tmp_path / "test_cache.json")

    memoria = Memoria(arquivo=db_path)
    cache = Cache(arquivo=cache_path)

    # Mock do ChromaDB para evitar dependência de embedding
    semantica_mock = MagicMock()
    semantica_mock.buscar_similar.return_value = []
    semantica_mock.adicionar.return_value = None
    semantica_mock.adicionar_conhecimento.return_value = None
    semantica_mock.estatisticas.return_value = {"documentos": 0, "embedding_model": "mock", "disponivel": False}

    sistema = SistemaAgentes(memoria=memoria, cache=cache, semantica=semantica_mock)
    yield sistema
    sistema.fechar()


class TestPipelineNivel1:
    """Testa resolução sem LLM (ferramentas, cache, ChromaDB)."""

    @patch("src.core.analisador.ollama")
    def test_ferramenta_calculo(self, mock_ollama, sistema_mock):
        """Cálculo resolve no nível 1 sem chamar LLM principal."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": (
                    '{"agente":"generalista","precisa_web":false,'
                    '"ferramenta":"calculo","parametros":{"expressao":"2+2"}}'
                )
            }
        }

        resultado = sistema_mock.executar("generalista", "quanto é 2+2")
        assert "4" in resultado

    @patch("src.core.analisador.ollama")
    def test_ferramenta_saudacao(self, mock_ollama, sistema_mock):
        """Saudação resolve no nível 1."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": '{"agente":"generalista","precisa_web":false,"ferramenta":"saudacao","parametros":{}}'
            }
        }

        resultado = sistema_mock.executar("generalista", "oi")
        assert "Olá" in resultado or "pronto" in resultado.lower()

    @patch("src.core.analisador.ollama")
    def test_ferramenta_data_hora(self, mock_ollama, sistema_mock):
        """Data/hora resolve no nível 1."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": '{"agente":"generalista","precisa_web":false,"ferramenta":"data_hora","parametros":{}}'
            }
        }

        resultado = sistema_mock.executar("generalista", "que horas são")
        assert "Hora:" in resultado or "Data:" in resultado

    @patch("src.core.analisador.ollama")
    def test_cache_hit_retorna_apos_classificar_frescor(self, mock_ollama, sistema_mock):
        """Cache hit só ocorre após confirmar que a pergunta não exige dados atuais."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": '{"agente":"generalista","precisa_web":false,"ferramenta":null,"parametros":{}}'
            }
        }

        # Popula cache manualmente
        sistema_mock.cache.salvar("generalista:como usar docker compose", "Use docker-compose up", "generalista")

        resultado = sistema_mock.executar("generalista", "como usar docker compose")
        assert resultado == "Use docker-compose up"
        mock_ollama.chat.assert_called_once()

    @patch("src.core.analisador.ollama")
    @patch("src.agentes.executor.pesquisar_web_rapida", return_value="dados novos")
    @patch("src.agentes.executor.chamar_llm")
    def test_cache_nao_antecipa_consulta_que_precisa_web(
        self, mock_llm, _mock_web, mock_ollama, sistema_mock
    ):
        sistema_mock.cache.salvar(
            "pesquisador:qual a versão do pacote example",
            "versão antiga",
            "pesquisador",
        )
        mock_ollama.chat.return_value = {
            "message": {
                "content": (
                    '{"agente":"pesquisador","precisa_web":true,'
                    '"ferramenta":null,"parametros":{}}'
                )
            }
        }
        mock_llm.return_value = {
            "resposta": "versão nova",
            "tempo_ms": 10,
            "tokens_entrada": 1,
            "tokens_saida": 1,
            "erro": False,
        }

        resultado = sistema_mock.executar(
            "generalista", "qual a versão do pacote example"
        )

        assert resultado == "versão nova"

    @patch("src.core.analisador.ollama")
    def test_ferramenta_deterministica_nao_carrega_coordenador(self, mock_ollama, sistema_mock):
        resultado = sistema_mock.executar("generalista", "quanto é 8*7")

        assert "56" in resultado
        assert sistema_mock.ultima_fonte == "ferramenta"
        mock_ollama.chat.assert_not_called()

    @patch("src.core.analisador.ollama")
    def test_acao_mutavel_exige_confirmacao(self, mock_ollama, sistema_mock):
        resultado = sistema_mock.executar(
            "generalista", "criar arquivo data/teste.txt ::: conteúdo"
        )

        assert "Confirmação necessária" in resultado
        assert sistema_mock.ultima_fonte == "confirmacao"
        assert sistema_mock.cache.buscar(
            "generalista:criar arquivo data/teste.txt ::: conteúdo"
        ) is None
        mock_ollama.chat.assert_not_called()


class TestPipelineNivel2:
    """Testa execução com modelo rápido."""

    @patch("src.agentes.executor.chamar_llm")
    @patch("src.core.analisador.ollama")
    def test_nivel2_retorna_resposta_llm(self, mock_analisador, mock_chamar_llm, sistema_mock):
        """Pergunta simples usa modelo rápido e retorna resposta."""
        mock_analisador.chat.return_value = {
            "message": {
                "content": '{"agente":"generalista","precisa_web":false,"ferramenta":null,"parametros":{}}'
            }
        }
        mock_chamar_llm.return_value = {
            "resposta": "Python é uma linguagem de programação interpretada.",
            "tempo_ms": 200,
            "tokens_entrada": 50,
            "tokens_saida": 20,
        }

        resultado = sistema_mock.executar("generalista", "o que é python")
        assert "Python" in resultado
        mock_chamar_llm.assert_called_once()

    @patch("src.agentes.executor.chamar_llm")
    @patch("src.agentes.executor.pesquisar_web_rapida")
    @patch("src.core.analisador.ollama")
    def test_nivel2_com_web(self, mock_analisador, mock_web, mock_chamar_llm, sistema_mock):
        """Quando precisa_web=true, busca web antes de chamar LLM."""
        mock_analisador.chat.return_value = {
            "message": {
                "content": '{"agente":"pesquisador","precisa_web":true,"ferramenta":null,"parametros":{}}'
            }
        }
        mock_web.return_value = "Node.js versão 22.0 é a mais recente."
        mock_chamar_llm.return_value = {
            "resposta": "A última versão do Node.js é a 22.0.",
            "tempo_ms": 300,
            "tokens_entrada": 80,
            "tokens_saida": 25,
        }

        resultado = sistema_mock.executar("generalista", "qual a última versão do node.js")
        mock_web.assert_called_once()
        assert "22" in resultado or "Node" in resultado


class TestPipelineNivel3:
    """Testa execução com modelo profundo + RAG."""

    @patch("src.agentes.executor.verificar_modelo_disponivel")
    @patch("src.agentes.executor.chamar_llm")
    @patch("src.agentes.executor.pesquisar_web_profunda")
    @patch("src.core.analisador.ollama")
    def test_nivel3_com_web_profunda(
        self, mock_analisador, mock_web, mock_chamar_llm, mock_modelo_disponivel, sistema_mock
    ):
        """Nível 3 usa web profunda e modelo profundo."""
        mock_analisador.chat.return_value = {
            "message": {
                "content": '{"agente":"programador","precisa_web":true,"ferramenta":null,"parametros":{}}'
            }
        }
        mock_web.return_value = "FastAPI é um framework moderno para APIs."
        mock_modelo_disponivel.return_value = True
        mock_chamar_llm.return_value = {
            "resposta": "Para criar uma API REST com JWT usando FastAPI...",
            "tempo_ms": 800,
            "tokens_entrada": 200,
            "tokens_saida": 150,
        }

        # Força nível 3 para ir direto ao pipeline profundo
        sistema_mock.forcar_nivel(3)
        resultado = sistema_mock.executar(
            "generalista", "crie uma API REST com autenticação JWT em Python usando FastAPI"
        )
        assert "FastAPI" in resultado or "API" in resultado


class TestRoteamento:
    """Testa roteamento automático de agente."""

    @patch("src.agentes.executor.chamar_llm")
    @patch("src.core.analisador.ollama")
    def test_redireciona_para_programador(self, mock_analisador, mock_chamar_llm, sistema_mock):
        """Analisador sugere programador, executor redireciona."""
        mock_analisador.chat.return_value = {
            "message": {
                "content": '{"agente":"programador","precisa_web":false,"ferramenta":null,"parametros":{}}'
            }
        }
        mock_chamar_llm.return_value = {
            "resposta": "def hello(): print('world')",
            "tempo_ms": 300,
            "tokens_entrada": 60,
            "tokens_saida": 30,
        }

        resultado = sistema_mock.executar("generalista", "crie uma função python simples")
        assert resultado  # Não vazio, não crashou


class TestNivelForcado:
    """Testa forcing de nível via forcar_nivel()."""

    @patch("src.agentes.executor.verificar_modelo_disponivel")
    @patch("src.agentes.executor.chamar_llm")
    @patch("src.core.analisador.ollama")
    def test_forcar_nivel3(self, mock_analisador, mock_chamar_llm, mock_modelo, sistema_mock):
        """forcar_nivel(3) faz usar modelo profundo mesmo para pergunta simples."""
        mock_analisador.chat.return_value = {
            "message": {
                "content": '{"agente":"generalista","precisa_web":false,"ferramenta":null,"parametros":{}}'
            }
        }
        mock_modelo.return_value = True
        mock_chamar_llm.return_value = {
            "resposta": "Resposta detalhada com modelo profundo.",
            "tempo_ms": 500,
            "tokens_entrada": 100,
            "tokens_saida": 80,
        }

        sistema_mock.forcar_nivel(3)
        resultado = sistema_mock.executar("generalista", "o que é REST")
        assert resultado == "Resposta detalhada com modelo profundo."


class TestErros:
    """Testa resiliência a erros."""

    @patch("src.core.analisador.ollama")
    def test_analisador_falha_usa_fallback(self, mock_ollama, sistema_mock):
        """Se o analisador falha, usa fallback conservador."""
        mock_ollama.chat.side_effect = Exception("Ollama offline")

        # Com fallback, deve cair em nível 1 → ferramentas
        resultado = sistema_mock.executar("generalista", "quanto é 5+5")
        assert "10" in resultado

    @patch("src.agentes.executor.chamar_llm")
    @patch("src.core.analisador.ollama")
    def test_resposta_erro_nao_e_cacheada(self, mock_analisador, mock_chamar_llm, sistema_mock):
        """Respostas de erro do LLM não são salvas no cache."""
        mock_analisador.chat.return_value = {
            "message": {
                "content": '{"agente":"generalista","precisa_web":false,"ferramenta":null,"parametros":{}}'
            }
        }
        mock_chamar_llm.return_value = {
            "resposta": "Erro ao chamar modelo qwen3:1.7b: connection refused",
            "tempo_ms": 100,
            "tokens_entrada": 0,
            "tokens_saida": 0,
        }

        sistema_mock.executar("generalista", "explique design patterns")
        assert sistema_mock.cache.buscar("generalista:explique design patterns") is None
