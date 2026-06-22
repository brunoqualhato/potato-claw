"""Testes para o pipeline de web RAG (Search → Fetch → Convert → Extract)."""

import pytest
from src.ferramentas.web_rag import html_para_markdown, ResultadoBusca


class TestHTMLParaMarkdown:
    """Testa conversão HTML → Markdown limpo."""

    def test_paragrafo_simples(self):
        html = "<p>Hello World</p>"
        md = html_para_markdown(html)
        assert "Hello World" in md

    def test_remove_script(self):
        html = "<p>Visível</p><script>var x = 1; alert('hack');</script><p>Também visível</p>"
        md = html_para_markdown(html)
        assert "Visível" in md
        assert "Também visível" in md
        assert "alert" not in md
        assert "var x" not in md

    def test_remove_style(self):
        html = "<style>.foo{color:red}</style><p>Conteúdo</p>"
        md = html_para_markdown(html)
        assert "Conteúdo" in md
        assert "color" not in md

    def test_remove_nav_footer(self):
        html = "<nav><a>Menu</a></nav><main><p>Conteúdo principal</p></main><footer>Rodapé</footer>"
        md = html_para_markdown(html)
        assert "Conteúdo principal" in md
        assert "Menu" not in md
        assert "Rodapé" not in md

    def test_headings(self):
        html = "<h1>Título</h1><h2>Sub</h2><h3>SubSub</h3>"
        md = html_para_markdown(html)
        assert "# Título" in md
        assert "## Sub" in md
        assert "### SubSub" in md

    def test_lista(self):
        html = "<ul><li>Item A</li><li>Item B</li></ul>"
        md = html_para_markdown(html)
        assert "- Item A" in md
        assert "- Item B" in md

    def test_codigo(self):
        html = "<pre>def foo():\n    return 42</pre>"
        md = html_para_markdown(html)
        assert "```" in md
        assert "def foo():" in md

    def test_truncamento(self):
        """Markdown grande é truncado."""
        html = "<p>" + "x" * 10000 + "</p>"
        md = html_para_markdown(html)
        assert len(md) <= 6100  # MAX_MD_CHARS + margem do indicador

    def test_html_vazio(self):
        md = html_para_markdown("")
        assert md == ""

    def test_html_mal_formado(self):
        """Não deve crashar com HTML quebrado."""
        html = "<p>Aberto sem fechar <div>outro <span>mix"
        md = html_para_markdown(html)
        assert "Aberto sem fechar" in md

    def test_tabela_preserva_conteudo(self):
        html = "<table><tr><td>Célula 1</td><td>Célula 2</td></tr></table>"
        md = html_para_markdown(html)
        assert "Célula 1" in md
        assert "Célula 2" in md

    def test_entidades_html(self):
        html = "<p>10 &gt; 5 &amp; 3 &lt; 8</p>"
        md = html_para_markdown(html)
        assert ">" in md or "&gt;" in md  # Parser converte automaticamente


class TestResultadoBusca:
    def test_dataclass(self):
        r = ResultadoBusca(titulo="Test", url="https://x.com", snippet="bla")
        assert r.titulo == "Test"
        assert r.url == "https://x.com"
