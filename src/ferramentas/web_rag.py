"""
Pipeline de pesquisa web profunda com grounding real.

Fluxo (inspirado em SearXNG mas otimizado para hardware fraco):
  1. SEARCH  → DuckDuckGo retorna URLs relevantes
  2. FETCH   → wget/httpx busca conteúdo real das páginas (top N)
  3. CONVERT → HTML → Markdown limpo (sem JS, nav, ads)
  4. EXTRACT → LLM pequena extrai APENAS fatos relevantes à pergunta
  5. GROUND  → LLM final responde usando SOMENTE os fatos extraídos

Benefícios vs snippet-only:
  - Conteúdo completo da página (não truncado em 200 chars)
  - Dados estruturados (tabelas, listas, código) preservados
  - LLM extratora remove ruído ANTES de chegar na LLM final
  - Reduz alucinação: resposta baseada em texto real, não em "memória"
  - Fontes citáveis com URL exata

Performance (Mac M1 8GB):
  - Fetch: ~200-800ms por página (paralelo)
  - Convert: ~5ms (regex, sem deps pesadas)
  - Extract: ~300-600ms (LLM 1.2B, prompt curto)
  - Total: ~1-2s para 3 páginas em paralelo
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Optional

from ddgs import DDGS

from src.core.config import (
    DATA_DIR,
    MODELOS,
    WEB_RAG_CACHE_TTL,
    WEB_RAG_FETCH_TIMEOUT,
    WEB_RAG_MAX_MD_CHARS,
    WEB_RAG_MAX_PAGINAS,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════

MAX_PAGINAS_FETCH = WEB_RAG_MAX_PAGINAS
FETCH_TIMEOUT_S = WEB_RAG_FETCH_TIMEOUT
MAX_HTML_BYTES = 300_000       # ~300KB máx por página (reduzido para poupar RAM em HW fraco)
MAX_MD_CHARS = WEB_RAG_MAX_MD_CHARS
MAX_CONTEXTO_TOTAL = 4000      # Chars totais enviados para a LLM extratora
MODELO_EXTRATOR = MODELOS["rapido"]  # LLM barata para extração
CACHE_FETCH_DIR = DATA_DIR / "web_cache"
CACHE_TTL_S = WEB_RAG_CACHE_TTL

# Headers realistas para evitar bloqueio
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}


# ══════════════════════════════════════════════════════════════
# 1. SEARCH — Busca URLs relevantes
# ══════════════════════════════════════════════════════════════


@dataclass(slots=True)
class ResultadoBusca:
    titulo: str
    url: str
    snippet: str


def buscar_urls(query: str, max_resultados: int = 5) -> list[ResultadoBusca]:
    """Retorna URLs rankeadas do DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_resultados))
        return [
            ResultadoBusca(
                titulo=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            )
            for r in raw
            if r.get("href")
        ]
    except Exception as e:
        logger.warning("Erro na busca DuckDuckGo: %s", e)
        return []


# ══════════════════════════════════════════════════════════════
# 2. FETCH — Baixa conteúdo real das páginas
# ══════════════════════════════════════════════════════════════


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _cache_valido(url: str) -> Optional[str]:
    """Retorna conteúdo cacheado se existir e não tiver expirado."""
    CACHE_FETCH_DIR.mkdir(parents=True, exist_ok=True)
    arquivo = CACHE_FETCH_DIR / f"{_url_hash(url)}.md"
    if arquivo.exists():
        idade = time.time() - arquivo.stat().st_mtime
        if idade < CACHE_TTL_S:
            return arquivo.read_text(encoding="utf-8")
    return None


def _salvar_cache(url: str, conteudo: str):
    """Salva markdown no cache local."""
    CACHE_FETCH_DIR.mkdir(parents=True, exist_ok=True)
    arquivo = CACHE_FETCH_DIR / f"{_url_hash(url)}.md"
    try:
        arquivo.write_text(conteudo, encoding="utf-8")
    except OSError as e:
        logger.debug("Erro ao salvar cache web para %s: %s", url, e)


def fetch_pagina(url: str) -> Optional[str]:
    """
    Baixa HTML de uma URL usando urllib (zero deps extras).
    Retorna HTML raw ou None em caso de erro.
    """
    # Checa cache primeiro
    cached = _cache_valido(url)
    if cached:
        return cached  # Já é markdown

    # Ignora URLs de arquivos pesados
    skip_extensions = (".pdf", ".zip", ".tar", ".gz", ".exe", ".dmg", ".mp4", ".mp3")
    if any(url.lower().endswith(ext) for ext in skip_extensions):
        return None

    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            # Verifica content-type
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None

            html = resp.read(MAX_HTML_BYTES).decode("utf-8", errors="ignore")
            return html
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, UnicodeDecodeError) as e:
        logger.debug("Fetch falhou para %s: %s", url, e)
        return None
    except Exception as e:
        logger.debug("Fetch erro inesperado para %s: %s", url, e)
        return None


def fetch_paginas_paralelo(urls: list[str], max_workers: int = 3) -> dict[str, str]:
    """Baixa múltiplas páginas em paralelo. Retorna {url: html}."""
    resultados = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_pagina, url): url for url in urls[:MAX_PAGINAS_FETCH]}
        for future in as_completed(futures, timeout=FETCH_TIMEOUT_S + 2):
            url = futures[future]
            try:
                html = future.result()
                if html:
                    resultados[url] = html
            except Exception:
                pass
    return resultados


# ══════════════════════════════════════════════════════════════
# 3. CONVERT — HTML → Markdown limpo
# ══════════════════════════════════════════════════════════════


class _HTMLParaTexto(HTMLParser):
    """Parser leve que extrai texto útil de HTML, ignorando navegação e scripts."""

    TAGS_BLOCO = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                  "li", "tr", "td", "th", "blockquote", "pre", "code",
                  "article", "section", "main"}
    TAGS_IGNORAR = {"script", "style", "nav", "footer", "header",
                    "aside", "noscript", "iframe", "svg", "form",
                    "button", "input", "select", "meta", "link"}
    TAGS_HEADING = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__()
        self._output: list[str] = []
        self._ignorar_depth = 0
        self._em_codigo = False
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list):
        tag = tag.lower()
        self._tag_stack.append(tag)

        if tag in self.TAGS_IGNORAR:
            self._ignorar_depth += 1
            return

        if self._ignorar_depth > 0:
            return

        if tag in self.TAGS_HEADING:
            nivel = int(tag[1])
            self._output.append(f"\n{'#' * nivel} ")
        elif tag == "li":
            self._output.append("\n- ")
        elif tag == "br":
            self._output.append("\n")
        elif tag == "pre" or tag == "code":
            self._em_codigo = True
            self._output.append("\n```\n")
        elif tag in self.TAGS_BLOCO:
            self._output.append("\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()

        if tag in self.TAGS_IGNORAR:
            self._ignorar_depth = max(0, self._ignorar_depth - 1)
            return

        if tag == "pre" or tag == "code":
            self._em_codigo = False
            self._output.append("\n```\n")
        elif tag in self.TAGS_BLOCO:
            self._output.append("\n")

    def handle_data(self, data: str):
        if self._ignorar_depth > 0:
            return
        texto = data if self._em_codigo else data.strip()
        if texto:
            self._output.append(texto)

    def get_markdown(self) -> str:
        return "".join(self._output)


def html_para_markdown(html: str) -> str:
    """Converte HTML para Markdown limpo — sem deps externas (~5ms)."""
    parser = _HTMLParaTexto()
    try:
        parser.feed(html)
    except Exception:
        # Fallback brutal: regex para remover tags
        texto = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        texto = re.sub(r'<style[^>]*>.*?</style>', '', texto, flags=re.DOTALL | re.IGNORECASE)
        texto = re.sub(r'<[^>]+>', ' ', texto)
        return re.sub(r'\s+', ' ', texto).strip()[:MAX_MD_CHARS]

    md = parser.get_markdown()

    # Limpeza pós-processamento
    md = re.sub(r'\n{3,}', '\n\n', md)          # Remove excesso de linhas vazias
    md = re.sub(r'[ \t]+\n', '\n', md)          # Trailing whitespace
    md = re.sub(r'```\s*```', '', md)            # Blocos de código vazios
    md = md.strip()

    # Trunca para não explodir contexto
    if len(md) > MAX_MD_CHARS:
        md = md[:MAX_MD_CHARS] + "\n\n[...conteúdo truncado]"

    return md


# ══════════════════════════════════════════════════════════════
# 4. EXTRACT — LLM pequena extrai fatos relevantes
# ══════════════════════════════════════════════════════════════


_PROMPT_EXTRATOR = """\
Você é um extrator de fatos. Sua tarefa é ler o texto fonte abaixo e extrair \
APENAS as informações relevantes para responder a pergunta do usuário.

REGRAS:
- Extraia SOMENTE fatos presentes no texto fonte
- NÃO invente, NÃO adicione conhecimento próprio
- Mantenha números, datas, nomes e URLs exatos
- Se o texto não contém informação relevante, diga "SEM_INFO"
- Responda em formato de bullet points
- Máximo 8 bullets
"""


def extrair_fatos(
    pergunta: str,
    documentos: list[dict],
    modelo: str = MODELO_EXTRATOR,
) -> str:
    """
    Usa LLM pequena para extrair fatos relevantes dos documentos.
    Cada documento é {url, titulo, markdown}.

    A LLM extratora NÃO responde a pergunta — ela apenas filtra ruído.
    Isso permite que a LLM final trabalhe com dados limpos.
    """
    import ollama

    # Monta contexto compacto para a extratora
    contexto_docs = ""
    chars_total = 0
    for doc in documentos:
        bloco = f"\n--- Fonte: {doc['url']} ---\n{doc['markdown']}\n"
        if chars_total + len(bloco) > MAX_CONTEXTO_TOTAL:
            # Trunca o último doc para caber
            espaco = MAX_CONTEXTO_TOTAL - chars_total
            if espaco > 200:
                bloco = bloco[:espaco] + "\n[truncado]"
            else:
                break
        contexto_docs += bloco
        chars_total += len(bloco)

    if not contexto_docs.strip():
        return ""

    try:
        response = ollama.chat(
            model=modelo,
            messages=[
                {"role": "system", "content": _PROMPT_EXTRATOR},
                {
                    "role": "user",
                    "content": (
                        f"PERGUNTA: {pergunta}\n\n"
                        f"TEXTO FONTE:\n{contexto_docs}"
                    ),
                },
            ],
            options={"temperature": 0.1, "num_predict": 400},
        )
        resultado = response["message"]["content"].strip()

        if "SEM_INFO" in resultado.upper():
            return ""

        return resultado
    except Exception:
        # Fallback: retorna os primeiros chars do markdown bruto
        return contexto_docs[:1500]


# ══════════════════════════════════════════════════════════════
# 5. PIPELINE COMPLETO — Search → Fetch → Convert → Extract
# ══════════════════════════════════════════════════════════════


@dataclass(slots=True)
class ResultadoWebRAG:
    """Resultado do pipeline completo de web RAG."""
    fatos_extraidos: str
    fontes: list[str] = field(default_factory=list)
    tempo_ms: int = 0
    paginas_baixadas: int = 0
    usou_cache: bool = False


def pesquisar_e_extrair(
    pergunta: str,
    max_paginas: int = MAX_PAGINAS_FETCH,
    usar_extrator: bool = True,
) -> ResultadoWebRAG:
    """
    Pipeline completo: Search → Fetch → Convert → Extract.

    Args:
        pergunta: A pergunta do usuário
        max_paginas: Máximo de páginas para baixar
        usar_extrator: Se True, usa LLM extratora. Se False, retorna markdown bruto.

    Returns:
        ResultadoWebRAG com fatos extraídos e metadados.
    """
    inicio = time.time()

    # 1. SEARCH
    resultados_busca = buscar_urls(pergunta, max_resultados=max_paginas + 2)
    if not resultados_busca:
        return ResultadoWebRAG(fatos_extraidos="", tempo_ms=0)

    urls = [r.url for r in resultados_busca[:max_paginas]]

    # 2. FETCH (paralelo)
    paginas_html = fetch_paginas_paralelo(urls)
    if not paginas_html:
        # Fallback: usa snippets do DuckDuckGo
        snippet_text = "\n".join(
            f"- {r.titulo}: {r.snippet} (fonte: {r.url})"
            for r in resultados_busca
        )
        return ResultadoWebRAG(
            fatos_extraidos=snippet_text,
            fontes=[r.url for r in resultados_busca],
            tempo_ms=int((time.time() - inicio) * 1000),
        )

    # 3. CONVERT
    documentos = []
    for url, conteudo in paginas_html.items():
        # Se veio do cache, já é markdown
        if _cache_valido(url):
            md = conteudo
        else:
            md = html_para_markdown(conteudo)
            _salvar_cache(url, md)

        titulo = next(
            (r.titulo for r in resultados_busca if r.url == url),
            url,
        )
        if md.strip():
            documentos.append({"url": url, "titulo": titulo, "markdown": md})

    if not documentos:
        return ResultadoWebRAG(fatos_extraidos="", tempo_ms=int((time.time() - inicio) * 1000))

    # 4. EXTRACT
    if usar_extrator:
        fatos = extrair_fatos(pergunta, documentos)
    else:
        # Sem extrator: concatena markdown bruto (para nível 2 economizar tempo)
        fatos = "\n\n".join(
            f"**{d['titulo']}** ({d['url']}):\n{d['markdown'][:800]}"
            for d in documentos
        )[:MAX_CONTEXTO_TOTAL]

    # Adiciona referências
    fontes = [d["url"] for d in documentos]
    if fatos and fontes:
        refs = "\n\nFontes:\n" + "\n".join(f"- {url}" for url in fontes)
        fatos += refs

    tempo_ms = int((time.time() - inicio) * 1000)

    return ResultadoWebRAG(
        fatos_extraidos=fatos,
        fontes=fontes,
        tempo_ms=tempo_ms,
        paginas_baixadas=len(documentos),
    )


# ══════════════════════════════════════════════════════════════
# ATALHOS PARA O EXECUTOR
# ══════════════════════════════════════════════════════════════


def pesquisar_web_profunda(pergunta: str, max_paginas: int = 3) -> str:
    """
    Substitui pesquisar_web() com pipeline completo.
    Retorna string formatada pronta para injetar no contexto do LLM.
    """
    resultado = pesquisar_e_extrair(pergunta, max_paginas=max_paginas, usar_extrator=True)

    if not resultado.fatos_extraidos:
        # Fallback para busca simples
        from src.ferramentas.web import pesquisar_web
        return pesquisar_web(pergunta, max_resultados=5)

    return resultado.fatos_extraidos


def pesquisar_web_rapida(pergunta: str, max_paginas: int = 2) -> str:
    """
    Versão rápida sem LLM extratora — para nível 2.
    Faz fetch + convert mas retorna markdown bruto truncado.
    """
    resultado = pesquisar_e_extrair(pergunta, max_paginas=max_paginas, usar_extrator=False)

    if not resultado.fatos_extraidos:
        from src.ferramentas.web import pesquisar_web
        return pesquisar_web(pergunta, max_resultados=3)

    return resultado.fatos_extraidos
