"""
Testes para o módulo sessao_codigo (funções puras sem Ollama).
Cobre parsing, validação, truncamento e lógica de sessão.
"""

from src.agentes.sessao_codigo import (
    SessaoCodigo,
    StepPlano,
    _criar_scaffold_offline,
    _extrair_codigo,
    _parse_plano,
    _parse_validacao,
    _truncar_inteligente,
    _validar_projeto_gerado,
    _validar_sintaxe,
    metricas_qualidade,
)


class TestValidarSintaxe:
    """Validação de sintaxe Python via ast.parse."""

    def test_codigo_valido(self):
        valido, erro = _validar_sintaxe("def foo():\n    return 42", "main.py")
        assert valido is True
        assert erro == ""

    def test_codigo_invalido(self):
        valido, erro = _validar_sintaxe("def foo(\n    return 42", "main.py")
        assert valido is False
        assert "SyntaxError" in erro

    def test_nao_python_sempre_valido(self):
        valido, erro = _validar_sintaxe("invalid {{{{ python", "style.css")
        assert valido is True

    def test_classe_valida(self):
        codigo = "class Foo:\n    def __init__(self):\n        self.x = 1"
        valido, _ = _validar_sintaxe(codigo, "models.py")
        assert valido is True

    def test_import_valido(self):
        valido, _ = _validar_sintaxe("from os import path\nimport sys", "util.py")
        assert valido is True


class TestValidarProjetoGerado:
    def test_projeto_python_valido(self):
        sessao = SessaoCodigo(objetivo="teste", scratchpad={"main.py": "print('ok')\n"})
        assert _validar_projeto_gerado(sessao) == []

    def test_detecta_python_invalido(self):
        sessao = SessaoCodigo(
            objetivo="teste",
            plano=[StepPlano(1, "entrada", "main.py")],
            scratchpad={"main.py": "def quebrado(\n"},
        )
        assert _validar_projeto_gerado(sessao)

    def test_scaffold_offline_usa_template(self):
        sessao = _criar_scaffold_offline("CLI Python para gerenciar tarefas")

        assert sessao.concluida is False
        assert sessao.projeto_validado is False
        assert "main.py" in sessao.scratchpad
        assert _validar_projeto_gerado(sessao) == []


class TestExtrairCodigo:
    """Extração de blocos de código do markdown."""

    def test_bloco_python(self):
        texto = "Aqui:\n```python\ndef foo():\n    return 1\n```\nFim."
        assert "def foo():" in _extrair_codigo(texto)
        assert "```" not in _extrair_codigo(texto)

    def test_bloco_sem_lang(self):
        texto = "```\nprint('hello')\n```"
        assert "print('hello')" in _extrair_codigo(texto)

    def test_sem_bloco_retorna_texto(self):
        texto = "def foo(): pass"
        assert _extrair_codigo(texto) == "def foo(): pass"

    def test_bloco_javascript(self):
        texto = "```javascript\nconst x = 1;\n```"
        resultado = _extrair_codigo(texto)
        assert "const x = 1;" in resultado
        assert "javascript" not in resultado

    def test_multiplos_blocos_usa_primeiro(self):
        texto = "```python\nfoo()\n```\nTexto\n```python\nbar()\n```"
        resultado = _extrair_codigo(texto)
        assert "foo()" in resultado


class TestParsePlano:
    """Parsing do JSON de planejamento."""

    def test_json_valido(self):
        raw = '{"steps": [{"descricao": "Criar main", "arquivo": "main.py", "dependencias": []}]}'
        sessao = _parse_plano("teste", raw)
        assert len(sessao.plano) == 1
        assert sessao.plano[0].arquivo == "main.py"
        assert sessao.plano[0].descricao == "Criar main"

    def test_json_com_dependencias(self):
        raw = (
            '{"steps": [{"descricao": "A", "arquivo": "a.py", "dependencias": []},'
            ' {"descricao": "B", "arquivo": "b.py", "dependencias": ["a.py"]}]}'
        )
        sessao = _parse_plano("teste", raw)
        assert len(sessao.plano) == 2
        assert sessao.plano[1].dependencias == ["a.py"]

    def test_json_invalido_retorna_fallback(self):
        raw = "não é json nenhum"
        sessao = _parse_plano("objetivo fallback", raw)
        assert len(sessao.plano) == 1
        assert sessao.plano[0].arquivo == "main.py"

    def test_json_com_lixo_ao_redor(self):
        raw = 'Plano: {"steps": [{"descricao": "X", "arquivo": "x.py", "dependencias": []}]} etc'
        sessao = _parse_plano("teste", raw)
        assert len(sessao.plano) == 1
        assert sessao.plano[0].arquivo == "x.py"

    def test_json_sem_steps(self):
        raw = '{"outra_coisa": 123}'
        sessao = _parse_plano("fallback", raw)
        assert len(sessao.plano) == 1  # Fallback


class TestParseValidacao:
    """Parsing do JSON de validação."""

    def test_valido_true(self):
        raw = '{"valido": true, "problemas": [], "decisoes": ["usar FastAPI"]}'
        r = _parse_validacao(raw)
        assert r["valido"] is True
        assert r["decisoes"] == ["usar FastAPI"]

    def test_valido_false(self):
        raw = '{"valido": false, "problemas": ["falta import"], "decisoes": []}'
        r = _parse_validacao(raw)
        assert r["valido"] is False
        assert "falta import" in r["problemas"]

    def test_json_invalido_assume_valido(self):
        r = _parse_validacao("blablabla sem json")
        assert r["valido"] is True

    def test_json_com_lixo(self):
        raw = 'Análise: {"valido": true, "problemas": [], "decisoes": []} fim'
        r = _parse_validacao(raw)
        assert r["valido"] is True


class TestTruncarInteligente:
    """Truncamento que preserva assinaturas."""

    def test_codigo_curto_nao_trunca(self):
        codigo = "import os\ndef foo(): pass"
        assert _truncar_inteligente(codigo, 1000) == codigo

    def test_trunca_mantendo_imports(self):
        linhas = ["import os", "import sys"] + [f"x = {i}" for i in range(200)]
        codigo = "\n".join(linhas)
        resultado = _truncar_inteligente(codigo, 100)
        assert "import os" in resultado
        assert "import sys" in resultado
        # Deve ser menor que o original
        assert len(resultado) < len(codigo)

    def test_trunca_mantendo_defs(self):
        linhas = ["def foo():", "    x = 1"] + ["    " + f"y = {i}" for i in range(100)]
        codigo = "\n".join(linhas)
        resultado = _truncar_inteligente(codigo, 100)
        assert "def foo():" in resultado


class TestSessaoCodigo:
    """Lógica da dataclass SessaoCodigo."""

    def test_progresso(self):
        sessao = SessaoCodigo(
            objetivo="teste",
            plano=[
                StepPlano(numero=1, descricao="A", concluido=True),
                StepPlano(numero=2, descricao="B", concluido=False),
            ],
        )
        assert sessao.progresso == "1/2"

    def test_concluida(self):
        sessao = SessaoCodigo(
            objetivo="teste",
            plano=[StepPlano(numero=1, descricao="A", concluido=True)],
        )
        assert sessao.concluida is True

    def test_nao_concluida(self):
        sessao = SessaoCodigo(
            objetivo="teste",
            plano=[StepPlano(numero=1, descricao="A", concluido=False)],
        )
        assert sessao.concluida is False

    def test_step_pendente(self):
        sessao = SessaoCodigo(
            objetivo="teste",
            plano=[
                StepPlano(numero=1, descricao="A", concluido=True),
                StepPlano(numero=2, descricao="B", concluido=False),
            ],
        )
        assert sessao.step_pendente().numero == 2

    def test_snapshot_e_rollback(self):
        sessao = SessaoCodigo(
            objetivo="teste",
            plano=[StepPlano(numero=1, descricao="A", arquivo="main.py")],
            scratchpad={"main.py": "original"},
        )
        sessao.snapshot()
        sessao.scratchpad["main.py"] = "modificado"

        sessao.rollback()
        assert sessao.scratchpad["main.py"] == "original"

    def test_registrar_resultado(self):
        sessao = SessaoCodigo(
            objetivo="teste",
            plano=[StepPlano(numero=1, descricao="A", arquivo="app.py")],
        )
        step = sessao.plano[0]
        sessao.registrar_resultado(step, "print('hello')")
        assert step.concluido is True
        assert sessao.scratchpad["app.py"] == "print('hello')"


class TestMetricasQualidade:
    """Cálculo de métricas."""

    def test_metricas_basicas(self):
        sessao = SessaoCodigo(
            objetivo="teste",
            plano=[
                StepPlano(numero=1, descricao="A", concluido=True, tentativas=1),
                StepPlano(numero=2, descricao="B", concluido=True, tentativas=2),
                StepPlano(numero=3, descricao="C", pulado=True, tentativas=3),
            ],
            scratchpad={"a.py": "x" * 100, "b.py": "y" * 200},
            tempo_total_ms=3000,
        )
        m = metricas_qualidade(sessao)
        assert "2/3" in m["taxa_sucesso"]
        assert m["pulados"] == 1
        assert m["total_chars_gerados"] == 300
