"""Testes para a extração robusta de código gerado por LLM (sessao_codigo)."""

from src.agentes.sessao_codigo import _extrair_codigo


class TestExtrairCodigo:
    """Cobre os casos que modelos pequenos costumam quebrar."""

    def test_bloco_com_tag_python(self):
        texto = "Aqui está:\n```python\nprint('oi')\n```"
        assert _extrair_codigo(texto) == "print('oi')"

    def test_bloco_sem_tag(self):
        assert _extrair_codigo("```\nx = 1\n```") == "x = 1"

    def test_sem_cerca_retorna_texto_cru(self):
        texto = "def f():\n    return 1"
        assert _extrair_codigo(texto) == "def f():\n    return 1"

    def test_multiplos_blocos_pega_o_maior(self):
        # Exemplo curto na explicação + código real (maior) depois.
        # A versão antiga pegava o primeiro (errado).
        texto = (
            "Exemplo rápido:\n```py\nx\n```\n"
            "Agora o código completo:\n```python\n"
            "def soma(a, b):\n    return a + b\n```"
        )
        assert _extrair_codigo(texto) == "def soma(a, b):\n    return a + b"

    def test_cerca_aberta_sem_fechamento(self):
        # Modelo esqueceu o ``` final.
        texto = "Segue:\n```python\nimport os\nprint(os.getcwd())"
        assert _extrair_codigo(texto) == "import os\nprint(os.getcwd())"

    def test_nao_come_primeira_linha_de_codigo(self):
        # Regressão: a primeira linha de código não pode ser confundida com tag.
        texto = "```python\nimport sys\nsys.exit(0)\n```"
        assert _extrair_codigo(texto) == "import sys\nsys.exit(0)"

    def test_prosa_antes_e_depois(self):
        texto = "blá blá\n```python\na = 1\n```\nfim do texto"
        assert _extrair_codigo(texto) == "a = 1"

    def test_tags_py_e_python3(self):
        assert _extrair_codigo("```py\nn = 2\n```") == "n = 2"
        assert _extrair_codigo("```python3\nn = 3\n```") == "n = 3"

    def test_texto_vazio(self):
        assert _extrair_codigo("") == ""
