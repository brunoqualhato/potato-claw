"""Testes para utilitários."""

from src.core.utils import normalizar


class TestNormalizar:
    def test_remove_acentos(self):
        assert normalizar("última versão") == "ultima versao"

    def test_lowercase(self):
        assert normalizar("HELLO WORLD") == "hello world"

    def test_combinado(self):
        assert normalizar("Documentação Técnica") == "documentacao tecnica"

    def test_string_vazia(self):
        assert normalizar("") == ""

    def test_sem_acentos_mantem(self):
        assert normalizar("python code") == "python code"
