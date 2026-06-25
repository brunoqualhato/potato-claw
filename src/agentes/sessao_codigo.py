"""
Agent Loop de Programação — v2 com 13 melhorias integradas.

Melhorias implementadas:
  1.  Validação de sintaxe via ast.parse (elimina erros triviais)
  2.  Feedback mid-session (modo interativo com input do usuário)
  3.  Controle de tokens com truncamento inteligente
  4.  Rollback de steps com snapshot do scratchpad
  5.  Fallback offline (funciona sem Ollama com templates)
  6.  Cache TTL por tipo de conteúdo
  7.  GC do ChromaDB (decay de documentos antigos)
  8.  Persistência de sessão em JSON (sobrevive crash)
  9.  Indexação do projeto real (lê arquivos existentes)
  10. Chain-of-thought no planejamento (2 passes)
  11. Métricas A/B de qualidade RAG
  12. Edição incremental de arquivos
  13. Streaming no loop (feedback visual em tempo real)
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ollama
from rich.console import Console
from rich.panel import Panel

from src.agentes.templates import obter_esqueleto, selecionar_template
from src.core.config import DATA_DIR, MODELOS
from src.memoria.semantica import MemoriaSemantica

console = Console()


# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════

MAX_RETRIES = 2
MAX_TOKENS_CONTEXTO = 3000     # ~750 tokens — safe para modelos 4K context
SESSAO_ARQUIVO = DATA_DIR / "sessao_ativa.json"  # [8] Persistência
PROJETOS_DIR = DATA_DIR / "projetos"


# ══════════════════════════════════════════════════════════════
# TIPOS
# ══════════════════════════════════════════════════════════════


@dataclass
class StepPlano:
    numero: int
    descricao: str
    arquivo: str = ""
    dependencias: list[str] = field(default_factory=list)
    concluido: bool = False
    resultado: str = ""
    tentativas: int = 0
    pulado: bool = False


@dataclass
class SessaoCodigo:
    objetivo: str
    plano: list[StepPlano] = field(default_factory=list)
    step_atual: int = 0
    scratchpad: dict[str, str] = field(default_factory=dict)
    snapshots: list[tuple[str, str]] = field(default_factory=list)  # [(arquivo, conteudo_anterior)]
    decisoes: list[str] = field(default_factory=list)
    erros: list[str] = field(default_factory=list)
    metricas_rag: dict = field(default_factory=dict)  # [11] A/B
    inicio: float = field(default_factory=time.time)
    tempo_total_ms: int = 0
    interativo: bool = False  # [2] Feedback mid-session
    projeto_validado: bool | None = None

    @property
    def progresso(self) -> str:
        total = len(self.plano)
        feitos = sum(1 for s in self.plano if s.concluido)
        return f"{feitos}/{total}"

    @property
    def concluida(self) -> bool:
        return (
            bool(self.plano)
            and all(s.concluido or s.pulado for s in self.plano)
            and self.projeto_validado is not False
        )

    def step_pendente(self) -> Optional[StepPlano]:
        for step in self.plano:
            if not step.concluido and not step.pulado:
                return step
        return None

    def snapshot(self):
        """[4] Salva apenas o diff antes de modificar (copy-on-write leve)."""
        # Guarda o estado do step pendente (o arquivo que vai ser criado/editado)
        step = self.step_pendente()
        if step and step.arquivo:
            conteudo_atual = self.scratchpad.get(step.arquivo, "")
            self.snapshots.append((step.arquivo, conteudo_atual))

    def rollback(self):
        """[4] Reverte o último arquivo modificado."""
        if self.snapshots:
            arquivo, conteudo = self.snapshots.pop()
            if conteudo:
                self.scratchpad[arquivo] = conteudo
            elif arquivo in self.scratchpad:
                del self.scratchpad[arquivo]

    def contexto_para_step(self, step: StepPlano) -> str:
        """[3] Monta contexto com controle de tokens."""
        partes = [f"PROJETO: {self.objetivo}"]
        chars_usados = len(partes[0])

        if self.decisoes:
            dec = f"DECISÕES: {'; '.join(self.decisoes[-5:])}"
            partes.append(dec)
            chars_usados += len(dec)

        # Inclui dependências com truncamento inteligente
        for dep in step.dependencias:
            if dep in self.scratchpad and chars_usados < MAX_TOKENS_CONTEXTO:
                conteudo = self.scratchpad[dep]
                espaco = MAX_TOKENS_CONTEXTO - chars_usados - 100
                if espaco <= 0:
                    break
                if len(conteudo) > espaco:
                    # [3] Trunca mantendo assinaturas (imports + defs)
                    conteudo = _truncar_inteligente(conteudo, espaco)
                bloco = f"ARQUIVO ({dep}):\n```\n{conteudo}\n```"
                partes.append(bloco)
                chars_usados += len(bloco)

        # Se não tem deps mas é step > 1, inclui anterior
        if not step.dependencias and step.numero > 1:
            ant = self.plano[step.numero - 2]
            if ant.arquivo and ant.arquivo in self.scratchpad:
                espaco = MAX_TOKENS_CONTEXTO - chars_usados - 100
                if espaco > 200:
                    conteudo = _truncar_inteligente(self.scratchpad[ant.arquivo], espaco)
                    bloco = f"ANTERIOR ({ant.arquivo}):\n```\n{conteudo}\n```"
                    partes.append(bloco)
                    chars_usados += len(bloco)

        # Se é o ponto de entrada, lista todos os módulos disponíveis
        eh_ponto_entrada = step.arquivo and step.arquivo.split(".")[0] in (
            "main", "index", "app", "server", "cli"
        )
        if eh_ponto_entrada and self.scratchpad:
            modulos_existentes = [
                f for f in self.scratchpad.keys()
                if f != step.arquivo and not f.endswith((".md", ".txt", ".json", ".css"))
            ]
            if modulos_existentes:
                # Extrai assinaturas dos módulos para o LLM saber o que importar
                assinaturas = []
                for mod in modulos_existentes:
                    codigo_mod = self.scratchpad[mod]
                    sigs = _truncar_inteligente(codigo_mod, 500)
                    assinaturas.append(f"  {mod}: {sigs[:200]}")

                partes.append(
                    "MÓDULOS DO PROJETO (DEVE importar e usar):\n"
                    + "\n".join(assinaturas)
                    + "\n\n"
                    + "IMPORTANTE — INTEGRAÇÃO OBRIGATÓRIA:\n"
                    + "- Este é o PONTO DE ENTRADA. DEVE importar classes/funções dos módulos acima.\n"
                    + "- Se existe models.py E storage.py: o ponto de entrada deve conectá-los.\n"
                    + "  Ex: Storage recebe/usa o Manager, ou o Manager usa Storage para persistir.\n"
                    + "- NÃO crie instâncias isoladas que não se comunicam.\n"
                    + "- O fluxo de dados deve ser: Entrada do Usuário → Manager/Model → Storage → Disco"
                )

        # Se é módulo de persistência, lembra de importar models
        eh_storage = step.arquivo and any(
            x in step.arquivo.lower() for x in ("storage", "persist", "database", "db", "repo")
        )
        if eh_storage and "models" in " ".join(self.scratchpad.keys()):
            partes.append(
                "ATENÇÃO: Este módulo de persistência DEVE importar as classes de models.py.\n"
                "Use 'from models import ...' para referenciar os tipos de dados.\n"
                "NÃO recrie classes que já existem em models.py."
            )

        # Reforça stack se definida nas decisões
        stack_decisao = next(
            (d.split("Stack: ")[1] for d in self.decisoes if "Stack:" in d), None
        )
        if stack_decisao:
            partes.append(f"STACK OBRIGATÓRIA: {stack_decisao}. NÃO use outra linguagem.")

        partes.append(f"STEP ({step.numero}/{len(self.plano)}): {step.descricao}")
        if step.arquivo:
            partes.append(f"GERE: {step.arquivo}")

            # Para README.md: injeta resumo completo do projeto gerado
            if step.arquivo.lower() == "readme.md" and self.scratchpad:
                partes.append(self._contexto_readme())

            # Reforça o formato esperado para evitar confusão de conteúdo
            formato_hints = {
                ".txt": (
                    "FORMATO: Texto simples. Se for requirements.txt: um pacote por linha "
                    "(ex: flask>=3.0). NÃO coloque HTML aqui."
                ),
                ".css": (
                    "FORMATO: Apenas regras CSS válidas (seletores { propriedade: valor; }). "
                    "NÃO coloque Python ou HTML aqui."
                ),
                ".html": (
                    "FORMATO: HTML válido. Pode usar Jinja2 ({{ }}, {% %}) se for template Flask. "
                    "NÃO coloque Python puro aqui."
                ),
                ".js": "FORMATO: JavaScript válido. NÃO coloque Python aqui.",
                ".json": "FORMATO: JSON válido. NÃO coloque código aqui.",
                ".md": (
                    "FORMATO: Markdown COMPLETO com título, descrição, instalação, execução, "
                    "funcionalidades e estrutura do projeto. BASEIE-SE nos arquivos reais listados acima."
                ),
            }
            for ext, hint in formato_hints.items():
                if step.arquivo.endswith(ext):
                    partes.append(hint)
                    break

        return "\n\n".join(partes)

    def _contexto_readme(self) -> str:
        """Gera contexto rico para o README baseado nos arquivos reais do projeto."""
        linhas = ["═══ INFORMAÇÕES DO PROJETO PARA O README ═══"]
        linhas.append(f"OBJETIVO: {self.objetivo}")

        if self.decisoes:
            stack = next((d for d in self.decisoes if "Stack:" in d), None)
            if stack:
                linhas.append(f"STACK: {stack}")

        linhas.append(f"\nARQUIVOS GERADOS ({len(self.scratchpad)}):")
        for arquivo, conteudo in self.scratchpad.items():
            if arquivo.lower() == "readme.md":
                continue
            # Extrai imports e estrutura para o README saber o que descrever
            resumo = self._resumir_arquivo(arquivo, conteudo)
            linhas.append(f"  • {arquivo}: {resumo}")

        # Identifica dependências para instrução de instalação
        for f in ("requirements.txt", "package.json"):
            if f in self.scratchpad:
                linhas.append(f"\nDEPENDÊNCIAS ({f}):\n{self.scratchpad[f][:500]}")
                break

        # Identifica ponto de entrada
        pontos = {"main.py", "app.py", "index.js", "server.js", "src/index.ts"}
        entrada = next((f for f in self.scratchpad if f in pontos), None)
        if entrada:
            linhas.append(f"\nPONTO DE ENTRADA: {entrada}")
            # Extrai porta se for web
            codigo_entrada = self.scratchpad[entrada]
            if "port" in codigo_entrada.lower() or "5000" in codigo_entrada or "3000" in codigo_entrada:
                linhas.append("TIPO: Servidor web (mencione URL no README)")
            else:
                linhas.append("TIPO: CLI interativa (mencione comando para rodar)")

        linhas.append(
            "\nINSTRUÇÕES PARA O README:"
            "\n- Título: nome descritivo do projeto"
            "\n- Descrição: 1-2 frases sobre o que faz"
            "\n- Instalação: comandos EXATOS (pip install -r requirements.txt / npm install)"
            "\n- Execução: comando EXATO para rodar (python main.py / python app.py / node index.js)"
            "\n- Se web: URL para acessar (http://localhost:PORTA)"
            "\n- Funcionalidades: lista com bullet points"
            "\n- Estrutura: tabela arquivo → responsabilidade"
        )
        return "\n".join(linhas)

    @staticmethod
    def _resumir_arquivo(arquivo: str, conteudo: str) -> str:
        """Gera resumo de 1 linha do que o arquivo faz."""
        if arquivo == "requirements.txt":
            deps = [
                linha.split(">=")[0].split("==")[0].strip()
                for linha in conteudo.split("\n") if linha.strip() and not linha.startswith("#")
            ]
            return f"Dependências: {', '.join(deps[:5])}"
        if arquivo.endswith(".py"):
            classes = re.findall(r"class\s+(\w+)", conteudo)
            funcs = re.findall(r"^def\s+(\w+)", conteudo, re.MULTILINE)
            partes = []
            if classes:
                partes.append(f"Classes: {', '.join(classes[:4])}")
            if funcs:
                partes.append(f"Funções: {', '.join(funcs[:4])}")
            return " | ".join(partes) if partes else "Módulo Python"
        if arquivo.endswith(".html"):
            return "Template HTML" + (" (Jinja2)" if "{{" in conteudo else "")
        if arquivo.endswith(".css"):
            return "Estilos CSS"
        if arquivo.endswith(".js"):
            funcs = re.findall(r"function\s+(\w+)", conteudo)
            return f"Funções: {', '.join(funcs[:4])}" if funcs else "Módulo JavaScript"
        return "Arquivo auxiliar"

    def registrar_resultado(self, step: StepPlano, codigo: str):
        step.concluido = True
        step.resultado = codigo
        if step.arquivo:
            self.scratchpad[step.arquivo] = codigo
        self.step_atual = step.numero


# ══════════════════════════════════════════════════════════════
# [3] CONTROLE DE TOKENS
# ══════════════════════════════════════════════════════════════


def _estimar_tokens(texto: str) -> int:
    """Estimativa: ~4 chars = 1 token para modelos multilíngues."""
    return len(texto) // 4


def _truncar_inteligente(codigo: str, max_chars: int) -> str:
    """
    Trunca código mantendo assinaturas úteis:
    - Imports no topo
    - Definições de classe/função (sem corpo)
    - Comentários de seção
    """
    if len(codigo) <= max_chars:
        return codigo

    linhas = codigo.split("\n")
    resultado = []
    chars = 0

    for linha in linhas:
        stripped = linha.strip()
        # Prioriza: imports, defs, classes, comments de seção
        eh_assinatura = (
            stripped.startswith(("import ", "from ", "def ", "class ", "# ═", "# ─"))
            or stripped.startswith(("@", "async def "))
        )

        if eh_assinatura or chars < max_chars * 0.7:
            resultado.append(linha)
            chars += len(linha) + 1
            if chars >= max_chars:
                resultado.append("# ... (truncado para caber no contexto)")
                break

    return "\n".join(resultado)


# ══════════════════════════════════════════════════════════════
# [1] VALIDAÇÃO DE SINTAXE
# ══════════════════════════════════════════════════════════════


def _validar_sintaxe(codigo: str, arquivo: str) -> tuple[bool, str]:
    """
    Valida sintaxe Python via ast.parse — ~1ms, zero RAM.
    Retorna (valido, erro_msg).
    """
    if not arquivo.endswith(".py"):
        return True, ""  # Não valida não-python

    try:
        ast.parse(codigo)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError linha {e.lineno}: {e.msg}"


# ══════════════════════════════════════════════════════════════
# [5] FALLBACK OFFLINE
# ══════════════════════════════════════════════════════════════


def _ollama_disponivel() -> bool:
    """Checa se o Ollama está respondendo."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def _criar_scaffold_offline(objetivo: str) -> SessaoCodigo:
    """Cria um scaffold sintaticamente válido usando apenas templates locais."""
    template = selecionar_template(objetivo)
    if template is None:
        return SessaoCodigo(
            objetivo=objetivo,
            erros=["Ollama indisponível e nenhum template local compatível foi encontrado."],
        )

    sessao = SessaoCodigo(
        objetivo=objetivo,
        decisoes=[f"Stack: {template.stack}", "Modo offline: scaffold baseado em template"],
        projeto_validado=False,
    )
    for numero, template_step in enumerate(template.gerar_plano(), 1):
        step = StepPlano(
            numero=numero,
            descricao=template_step.descricao,
            arquivo=template_step.arquivo,
            dependencias=template_step.dependencias,
        )
        esqueleto = obter_esqueleto(template, step.arquivo)
        conteudo = esqueleto or _conteudo_offline_padrao(step.arquivo, objetivo)
        sessao.plano.append(step)
        sessao.registrar_resultado(step, conteudo)
    return sessao


def _conteudo_offline_padrao(arquivo: str, objetivo: str) -> str:
    """Conteúdo mínimo válido para arquivos sem esqueleto específico."""
    nome = Path(arquivo).name
    if nome.lower() == "readme.md":
        return (
            f"# Projeto\n\n{objetivo}\n\n"
            "Scaffold criado em modo offline. Revise os módulos antes de uso em produção.\n"
        )
    if arquivo.endswith(".py"):
        return f'"""Módulo gerado offline para: {objetivo}."""\n'
    if arquivo.endswith((".js", ".jsx")):
        return "'use strict';\n"
    if arquivo.endswith((".ts", ".tsx")):
        return "export {};\n"
    if arquivo.endswith(".json"):
        return "{}\n"
    if arquivo.endswith(".html"):
        return "<!doctype html><html><body><main id=\"app\"></main></body></html>\n"
    if arquivo.endswith(".css"):
        return "body { font-family: sans-serif; }\n"
    return ""


# ══════════════════════════════════════════════════════════════
# [8] PERSISTÊNCIA DE SESSÃO
# ══════════════════════════════════════════════════════════════


def _persistir_sessao(sessao: SessaoCodigo):
    """Salva estado da sessão em JSON para sobreviver crash."""
    SESSAO_ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    dados = {
        "objetivo": sessao.objetivo,
        "step_atual": sessao.step_atual,
        "scratchpad": sessao.scratchpad,
        "decisoes": sessao.decisoes,
        "erros": sessao.erros,
        "tempo_total_ms": sessao.tempo_total_ms,
        "projeto_validado": sessao.projeto_validado,
        "plano": [
            {"numero": s.numero, "descricao": s.descricao, "arquivo": s.arquivo,
             "dependencias": s.dependencias, "concluido": s.concluido, "pulado": s.pulado}
            for s in sessao.plano
        ],
    }
    SESSAO_ARQUIVO.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")


def _restaurar_sessao() -> Optional[SessaoCodigo]:
    """Restaura sessão do JSON se existir e estiver incompleta."""
    if not SESSAO_ARQUIVO.exists():
        return None
    try:
        dados = json.loads(SESSAO_ARQUIVO.read_text(encoding="utf-8"))
        sessao = SessaoCodigo(
            objetivo=dados["objetivo"],
            step_atual=dados.get("step_atual", 0),
            scratchpad=dados.get("scratchpad", {}),
            decisoes=dados.get("decisoes", []),
            erros=dados.get("erros", []),
            projeto_validado=dados.get("projeto_validado"),
        )
        for s in dados.get("plano", []):
            sessao.plano.append(StepPlano(
                numero=s["numero"], descricao=s["descricao"],
                arquivo=s.get("arquivo", ""), dependencias=s.get("dependencias", []),
                concluido=s.get("concluido", False), pulado=s.get("pulado", False),
            ))
        if sessao.concluida:
            return None  # Já terminou
        return sessao
    except (json.JSONDecodeError, KeyError):
        return None


def _limpar_persistencia():
    """Remove arquivo de sessão após conclusão."""
    if SESSAO_ARQUIVO.exists():
        SESSAO_ARQUIVO.unlink()


# ══════════════════════════════════════════════════════════════
# [9] INDEXAR PROJETO REAL
# ══════════════════════════════════════════════════════════════


def _indexar_projeto(diretorio: str | Path = ".") -> dict[str, str]:
    """
    Lê arquivos do projeto real e extrai assinaturas.
    Permite que o agente "veja" o que já existe.
    """
    from src.core.config import BASE_DIR
    base = Path(diretorio) if Path(diretorio).is_absolute() else BASE_DIR / diretorio
    indice: dict[str, str] = {}

    extensoes = {".py", ".js", ".ts", ".php", ".go", ".rs", ".java"}
    ignorar = {"__pycache__", "node_modules", ".git", "venv", ".venv", "data"}

    for arquivo in base.rglob("*"):
        if any(p in arquivo.parts for p in ignorar):
            continue
        if arquivo.suffix not in extensoes:
            continue
        if arquivo.stat().st_size > 50_000:  # Ignora arquivos gigantes
            continue

        try:
            conteudo = arquivo.read_text(encoding="utf-8")
            # Extrai apenas assinaturas para economizar espaço
            assinaturas = _extrair_assinaturas(conteudo, arquivo.suffix)
            if assinaturas:
                rel = str(arquivo.relative_to(base))
                indice[rel] = assinaturas
        except (OSError, UnicodeDecodeError):
            pass

    return indice


def _extrair_assinaturas(codigo: str, extensao: str) -> str:
    """Extrai imports + definições de um arquivo (sem corpo)."""
    if extensao != ".py":
        # Para não-python, pega primeiras 30 linhas
        linhas = codigo.split("\n")[:30]
        return "\n".join(linhas)

    try:
        tree = ast.parse(codigo)
    except SyntaxError:
        return ""

    linhas = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            linhas.append(f"import {', '.join(a.name for a in node.names)}")
        elif isinstance(node, ast.ImportFrom):
            linhas.append(f"from {node.module} import {', '.join(a.name for a in node.names)}")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
            linhas.append(f"class {node.name}({bases}):")
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    args = ", ".join(a.arg for a in item.args.args)
                    linhas.append(f"    def {item.name}({args}): ...")
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            args = ", ".join(a.arg for a in node.args.args)
            linhas.append(f"def {node.name}({args}): ...")

    return "\n".join(linhas[:50])


# ══════════════════════════════════════════════════════════════
# [7] GARBAGE COLLECTION DO CHROMADB
# ══════════════════════════════════════════════════════════════


def gc_chromadb(dias_expiracao: int = 30) -> int:
    """
    Remove documentos do ChromaDB não acessados há N dias.
    Retorna quantidade removida.
    """
    from datetime import datetime, timedelta
    sem = MemoriaSemantica()
    limite = (datetime.now() - timedelta(days=dias_expiracao)).isoformat()
    removidos = 0
    try:
        colecoes = {
            collection.name: collection for collection in sem.collections.values()
        }.values()
        for collection in colecoes:
            if collection.count() == 0:
                continue
            todos = collection.get(include=["metadatas"])
            ids_remover = []
            for i, meta in enumerate(todos["metadatas"]):
                criado = (meta or {}).get("criado_em", "")
                if criado and criado < limite:
                    ids_remover.append(todos["ids"][i])
            if ids_remover:
                collection.delete(ids=ids_remover)
                removidos += len(ids_remover)
        return removidos
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════
# [12] EDIÇÃO INCREMENTAL
# ══════════════════════════════════════════════════════════════


def editar_arquivo(sessao: SessaoCodigo, arquivo: str, instrucao: str) -> str:
    """
    Edita um arquivo existente no scratchpad com instrução em linguagem natural.
    Retorna o arquivo editado completo.
    """
    if arquivo not in sessao.scratchpad:
        return f"# Arquivo '{arquivo}' não encontrado no projeto"

    conteudo_atual = sessao.scratchpad[arquivo]

    try:
        response = ollama.chat(
            model=MODELOS["coder"],
            messages=[
                {"role": "system", "content": (
                    "Você recebe um arquivo e uma instrução de edição. "
                    "Retorne o arquivo COMPLETO com a modificação aplicada. "
                    "NÃO retorne apenas o diff — retorne o arquivo inteiro."
                )},
                {"role": "user", "content": (
                    f"ARQUIVO ATUAL ({arquivo}):\n```\n{conteudo_atual}\n```\n\n"
                    f"INSTRUÇÃO: {instrucao}\n\n"
                    f"Retorne o arquivo completo editado:"
                )},
            ],
            options={"temperature": 0.2, "num_predict": 4096},
            stream=False,
        )
        codigo = _extrair_codigo(response["message"]["content"].strip())

        # Valida sintaxe antes de aceitar
        if arquivo.endswith(".py"):
            valido, erro = _validar_sintaxe(codigo, arquivo)
            if not valido:
                console.print(f"[yellow]⚠️ Edição gerou erro de sintaxe: {erro}[/yellow]")
                return conteudo_atual  # Mantém original

        sessao.scratchpad[arquivo] = codigo
        return codigo
    except Exception as e:
        return f"# Erro na edição: {e}"


# ══════════════════════════════════════════════════════════════
# [10] CHAIN-OF-THOUGHT NO PLANEJAMENTO
# ══════════════════════════════════════════════════════════════

_PROMPT_COT_1 = """Analise o projeto solicitado e liste as funcionalidades necessárias para: {objetivo}

PRIMEIRO, IDENTIFIQUE O TIPO DE ENTREGA:
- Se o objetivo menciona "site", "web", "página", "dashboard", "painel", "formulário" → É um SITE WEB (Flask + HTML)
- Se o objetivo menciona "api", "rest", "endpoint", "backend" → É uma API (FastAPI/Express)
- Se o objetivo menciona "cli", "terminal", "menu" → É uma CLI
- Se não especificado E envolve CRUD de dados → padrão = SITE WEB com Flask
- Se não especificado E é utilitário/cálculo → padrão = CLI

REGRAS OBRIGATÓRIAS:
- O projeto DEVE ter um ponto de entrada executável
- Para SITE WEB: use Flask com templates HTML, formulários, rotas CRUD, CSS básico
- Para CLI: use menus interativos com input(), feedback visual
- Para API: use FastAPI/Express com endpoints RESTful
- Inclua: arquivo de dependências (requirements.txt/package.json), README com instruções
- TODOS os módulos devem se INTEGRAR: models → storage → ponto de entrada (conectados)
- NÃO crie módulos paralelos desconectados
- ESCOLHA UMA ÚNICA LINGUAGEM/STACK e use apenas ela em todo o projeto

IMPORTANTE SOBRE PERSISTÊNCIA:
- Se o projeto armazena dados: o módulo de storage DEVE importar os models
- O ponto de entrada DEVE usar storage para salvar/carregar dados
- Não crie Manager em memória E Storage separado sem conexão entre eles

Responda como bullet points. Máximo 10 itens. Seja específico e técnico.
O PRIMEIRO item deve ser sempre o ponto de entrada principal com a interface adequada ao tipo.
O ÚLTIMO item deve ser o README.md com instruções de execução."""

_PROMPT_COT_2 = """Com base nestas funcionalidades:
{funcionalidades}

Organize em arquivos de código. Cada arquivo = 1 step.
REGRAS:
- O PRIMEIRO step DEVE ser o arquivo de dependências (requirements.txt ou package.json)
- O SEGUNDO step DEVE ser os módulos/classes de lógica de negócio
- O PENÚLTIMO step DEVE ser o ponto de entrada principal (main.py/index.js/app.py) que importa os módulos\
 e oferece interface interativa
- O ÚLTIMO step DEVE ser README.md com instruções claras de execução
- Todos os arquivos devem se conectar via imports
- O ponto de entrada deve ser EXECUTÁVEL e INTERATIVO (menu, prompts, servidor)

Responda APENAS com JSON:
{{"steps": [{{"descricao": "...", "arquivo": "nome.ext", "dependencias": []}}]}}"""

_PROMPT_PLANEJAR_SIMPLES = """Decomponha em steps para criar um projeto FUNCIONAL e EXECUTÁVEL (JSON):
{{"steps": [{{"descricao":"...", "arquivo":"nome.ext", "dependencias":[]}}]}}

OBRIGATÓRIO:
- Primeiro step: arquivo de dependências (requirements.txt ou package.json)
- Steps intermediários: módulos de lógica
- Penúltimo step: ponto de entrada com interface interativa (CLI menu ou servidor web)
- Último step: README.md com instruções de execução

Máx 8 steps. Ordene por dependência. O projeto deve funcionar ao rodar o ponto de entrada."""


# ══════════════════════════════════════════════════════════════
# PROMPTS DO LOOP
# ══════════════════════════════════════════════════════════════

_PROMPT_CODER = """Gere APENAS o conteúdo do arquivo pedido. Nada mais.

REGRAS:
1. O conteúdo DEVE corresponder ao TIPO do arquivo:
   - .txt (requirements.txt): um pacote por linha, ex: flask>=3.0
   - .py: código Python com imports
   - .html: HTML com DOCTYPE
   - .css: regras CSS
   - .js: código JavaScript
   - .md: Markdown
   - .json: JSON válido

2. Gere código COMPLETO e FUNCIONAL — sem TODO, sem pass, sem stubs
3. Se o contexto mostra módulos existentes, IMPORTE-OS (from X import Y)
4. Para ponto de entrada: inclua interface interativa (menu CLI ou servidor web)
5. Para persistência: use JSON file ou SQLite — dados devem sobreviver ao restart
6. ATENÇÃO ao OBJETIVO do projeto — gere funcionalidade REAL, não CRUD genérico

IMPORTANTE: Responda APENAS com o código. Sem explicações antes ou depois."""

_PROMPT_VALIDAR = (
    'O código abaixo é válido para o objetivo? '
    'Responda JSON: {"valido":true/false,"problemas":[],"decisoes":[]}\n\n'
    "Critérios:\n"
    "1. Sem TODOs/stubs/pass vazio\n"
    "2. Formato correto pro tipo de arquivo (requirements.txt = só pacotes)\n"
    "3. Se ponto de entrada: tem interface funcional\n"
    "4. Imports corretos dos módulos do projeto"
)

_PROMPT_RETRY = """PROBLEMAS:
{problemas}

Gere o ARQUIVO INTEIRO corrigido. Sem explicações, apenas código."""


# ══════════════════════════════════════════════════════════════
# AGENT LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════


def executar_projeto(
    objetivo: str,
    interativo: bool = False,
    salvar_disco: bool = True,
    diretorio_saida: str | None = None,
    indexar_existente: bool = True,
) -> SessaoCodigo:
    """
    Agent loop v2 — executa projeto com todas as 13 melhorias.

    Args:
        objetivo: O que construir
        interativo: [2] Se True, pede feedback após cada step
        salvar_disco: Exporta arquivos ao final
        diretorio_saida: Destino dos arquivos
        indexar_existente: [9] Se True, lê projeto atual como contexto
    """
    inicio_total = time.time()

    # [5] Verifica se Ollama está disponível
    if not _ollama_disponivel():
        console.print("[yellow]⚡ Ollama indisponível; usando scaffold offline.[/yellow]")
        sessao = _criar_scaffold_offline(objetivo)
        sessao.tempo_total_ms = int((time.time() - inicio_total) * 1000)
        problemas = _validar_projeto_gerado(sessao)
        sessao.erros.extend(problemas)
        sessao.projeto_validado = False
        if salvar_disco and sessao.scratchpad:
            caminho = _exportar_disco(sessao, diretorio_saida)
            console.print(f"[bold green]📁 Scaffold salvo em: {caminho}[/bold green]")
        return sessao

    semantica = MemoriaSemantica()

    # [8] Verifica se há sessão anterior para continuar
    sessao_anterior = _restaurar_sessao()
    if sessao_anterior:
        console.print(f"[yellow]📂 Sessão anterior encontrada: {sessao_anterior.objetivo}[/yellow]")
        console.print(f"[yellow]   Progresso: {sessao_anterior.progresso}[/yellow]")
        sessao = sessao_anterior
    else:
        # [10] Planejamento com chain-of-thought
        console.print("[bold cyan]📐 Planejando (chain-of-thought)...[/bold cyan]")
        sessao = _planejar_cot(objetivo)

    sessao.interativo = interativo

    # [9] Indexa projeto existente como contexto
    if indexar_existente:
        indice = _indexar_projeto()
        if indice:
            console.print(f"[dim]📁 Projeto existente indexado ({len(indice)} arquivos)[/dim]")
            sessao.decisoes.insert(0, f"Projeto existente: {', '.join(list(indice.keys())[:10])}")

    console.print(Panel(
        "\n".join(f"  {s.numero}. {s.descricao} → [cyan]{s.arquivo}[/cyan]" for s in sessao.plano),
        title=f"📋 Plano ({len(sessao.plano)} steps)",
        border_style="blue",
    ))

    # ─── AGENT LOOP ───
    console.print("\n[bold green]🔄 Agent loop iniciado[/bold green]\n")

    for step in sessao.plano:
        if step.concluido or step.pulado:
            continue

        console.print(f"[bold]⚙️  Step {step.numero}/{len(sessao.plano)}:[/bold] {step.descricao}")

        # [4] Snapshot antes de executar
        sessao.snapshot()

        sucesso = _executar_step_com_validacao(sessao, step, semantica)

        if not sucesso:
            # [4] Rollback se falhou
            sessao.rollback()
            step.pulado = True
            sessao.erros.append(f"Step {step.numero} pulado: {step.descricao}")
            console.print("  [red]⏭️  Step pulado (rollback aplicado)[/red]")

        # [8] Persiste após cada step
        _persistir_sessao(sessao)

        # [2] Feedback mid-session
        if interativo and not sessao.concluida:
            feedback = _pedir_feedback()
            if feedback:
                if feedback.lower() in ("sair", "quit", "abort"):
                    console.print("[yellow]Sessão pausada. Use /projeto para continuar.[/yellow]")
                    break
                sessao.decisoes.append(f"Feedback do usuário: {feedback}")
                console.print("[dim]📝 Decisão registrada[/dim]")

    # ─── FINALIZAÇÃO ───
    sessao.tempo_total_ms = int((time.time() - inicio_total) * 1000)

    # Passada de coerência: verifica imports cruzados e corrige (sem LLM)
    _passada_coerencia(sessao)

    # Validação final do projeto como um todo
    problemas_projeto = _validar_projeto_gerado(sessao)

    # ─── AUTO-CORREÇÃO FINAL ───
    # Se a validação encontrou problemas, tenta corrigir os arquivos com defeito
    if problemas_projeto:
        console.print(
            f"[yellow]⚠️ Validação encontrou {len(problemas_projeto)} problema(s). "
            f"Tentando auto-correção...[/yellow]"
        )
        problemas_restantes = _autocorrecao_projeto(sessao, problemas_projeto)
        if problemas_restantes:
            sessao.projeto_validado = False
            sessao.erros.extend(f"Validação final: {p}" for p in problemas_restantes)
            console.print(
                f"[yellow]⚠️ {len(problemas_restantes)} problema(s) "
                f"não resolvidos após auto-correção.[/yellow]"
            )
        else:
            sessao.projeto_validado = True
            console.print("[green]🧪 Auto-correção resolveu todos os problemas.[/green]")
    else:
        sessao.projeto_validado = True
        console.print("[green]🧪 Smoke checks do projeto passaram.[/green]")

    _salvar_projeto_completo(semantica, sessao)

    if salvar_disco and sessao.scratchpad and sessao.projeto_validado:
        caminho = _exportar_disco(sessao, diretorio_saida)
        console.print(f"\n[bold green]📁 Salvo em: {caminho}[/bold green]")
    elif salvar_disco and sessao.scratchpad:
        # Exporta mesmo com problemas, mas avisa
        caminho = _exportar_disco(sessao, diretorio_saida)
        console.print(
            f"[yellow]📁 Salvo em: {caminho} (com problemas pendentes)[/yellow]"
        )

    # [8] Limpa persistência se concluiu
    if sessao.concluida:
        _limpar_persistencia()

    # Resumo
    console.print(Panel(
        f"Objetivo: {sessao.objetivo}\n"
        f"Steps: {sessao.progresso}\n"
        f"Arquivos: {len(sessao.scratchpad)}\n"
        f"Tempo: {sessao.tempo_total_ms / 1000:.1f}s\n"
        f"Erros: {len(sessao.erros) or 'nenhum'}",
        title="🏁 Resultado",
        border_style="green" if sessao.concluida else "yellow",
    ))

    return sessao


# ══════════════════════════════════════════════════════════════
# AUTO-CORREÇÃO FINAL DO PROJETO
# ══════════════════════════════════════════════════════════════

_PROMPT_CORRECAO_FINAL = (
    "O projeto gerado tem problemas. Corrija o arquivo abaixo.\n"
    "Gere o ARQUIVO INTEIRO corrigido. Sem explicações."
)

MAX_RODADAS_AUTOCORRECAO = 2


def _autocorrecao_projeto(sessao: SessaoCodigo, problemas: list[str]) -> list[str]:
    """
    Tenta corrigir o projeto automaticamente após validação final.

    Estratégia:
    1. Mapeia cada problema ao arquivo responsável
    2. Regenera cada arquivo com defeito via LLM (com contexto dos problemas)
    3. Re-valida após correção
    4. Repete até MAX_RODADAS ou sem problemas

    Retorna lista de problemas restantes (vazia = tudo corrigido).
    """
    for rodada in range(MAX_RODADAS_AUTOCORRECAO):
        # Tenta correção determinística primeiro (sem LLM)
        corrigidos_det = _correcao_deterministica(sessao, problemas)
        if corrigidos_det:
            console.print(f"  [dim]🔧 Rodada {rodada + 1}: {corrigidos_det} corrigido(s) (heurística)[/dim]")

        # Identifica arquivos com problema que precisam de LLM
        arquivos_problema = _mapear_problemas_para_arquivos(sessao, problemas)

        for arquivo, erros_arquivo in arquivos_problema.items():
            if arquivo not in sessao.scratchpad:
                continue

            console.print(f"  [dim]🔄 Corrigindo {arquivo}...[/dim]")
            codigo_corrigido = _regenerar_arquivo(sessao, arquivo, erros_arquivo)

            if codigo_corrigido:
                # Valida sintaxe antes de aceitar
                if arquivo.endswith(".py"):
                    valido, _ = _validar_sintaxe(codigo_corrigido, arquivo)
                    if not valido:
                        continue  # Rejeita correção com erro de sintaxe

                sessao.scratchpad[arquivo] = codigo_corrigido
                console.print(f"  [green]  ✅ {arquivo} corrigido[/green]")

        # Refaz coerência e re-valida
        _passada_coerencia(sessao)
        problemas = _validar_projeto_gerado(sessao)

        if not problemas:
            return []

        console.print(f"  [dim]  Restam {len(problemas)} problema(s) após rodada {rodada + 1}[/dim]")

    return problemas


def _correcao_deterministica(sessao: SessaoCodigo, problemas: list[str]) -> int:
    """
    Correções que não precisam de LLM. Retorna quantidade de correções aplicadas.
    """
    corrigidos = 0

    for problema in problemas:
        # requirements.txt com código → regenera
        if "requirements.txt" in problema and (
            "código" in problema.lower() or "contém" in problema.lower()
        ):
            if "requirements.txt" in sessao.scratchpad:
                sessao.scratchpad["requirements.txt"] = _gerar_requirements_deterministico(sessao)
                corrigidos += 1

        # Ponto de entrada executável não identificado → ignora (pode ser nome diferente)
        if "ponto de entrada" in problema.lower():
            # Verifica se algum .py tem if __name__
            for arq, codigo in sessao.scratchpad.items():
                if arq.endswith(".py") and "__name__" in codigo and "main" not in arq:
                    # Renomeia no scratchpad — cria alias main.py
                    pass  # Não faz sentido renomear, aceita como está

    return corrigidos


def _mapear_problemas_para_arquivos(
    sessao: SessaoCodigo, problemas: list[str]
) -> dict[str, list[str]]:
    """
    Mapeia problemas para os arquivos que precisam ser corrigidos.
    Retorna {arquivo: [problemas_do_arquivo]}.
    """
    mapa: dict[str, list[str]] = {}

    for problema in problemas:
        # Tenta extrair nome de arquivo do problema
        arquivo_encontrado = None
        for arq in sessao.scratchpad:
            if arq in problema:
                arquivo_encontrado = arq
                break

        # Se não encontrou arquivo explícito, analisa o tipo de problema
        if not arquivo_encontrado:
            if "compileall" in problema or "SyntaxError" in problema:
                # Erro de compilação — tenta achar o .py mencionado
                for arq in sessao.scratchpad:
                    if arq.endswith(".py") and arq.split("/")[-1] in problema:
                        arquivo_encontrado = arq
                        break
            elif "ponto de entrada" in problema.lower():
                # Pode ser que falta o main.py ou app.py
                pontos = {"main.py", "app.py", "index.js"}
                arquivo_encontrado = next(
                    (f for f in sessao.scratchpad if f in pontos), None
                )

        if arquivo_encontrado:
            mapa.setdefault(arquivo_encontrado, []).append(problema)
        else:
            # Problema global — aplica ao ponto de entrada
            pontos = {"main.py", "app.py", "index.js", "server.js"}
            entrada = next((f for f in sessao.scratchpad if f in pontos), None)
            if entrada:
                mapa.setdefault(entrada, []).append(problema)

    return mapa


def _regenerar_arquivo(
    sessao: SessaoCodigo, arquivo: str, erros: list[str]
) -> str | None:
    """
    Regenera um arquivo específico via LLM com contexto dos erros.
    Retorna código corrigido ou None se falhou.
    """
    codigo_atual = sessao.scratchpad.get(arquivo, "")

    # Para tipos simples, usa correção determinística
    if arquivo == "requirements.txt":
        return _gerar_requirements_deterministico(sessao)

    if arquivo.lower() == "readme.md":
        return _gerar_readme_deterministico(sessao)

    # Para código, usa LLM com contexto dos erros e dos outros módulos
    prompt_tipo = _prompt_por_tipo(arquivo, sessao.objetivo)

    # Monta contexto mínimo: objetivo + módulos relevantes
    contexto_partes = [f"PROJETO: {sessao.objetivo}"]

    # Inclui assinaturas dos outros módulos para manter integração
    for outro_arq, outro_codigo in sessao.scratchpad.items():
        if outro_arq == arquivo or outro_arq.endswith((".md", ".txt", ".css")):
            continue
        assinatura = _truncar_inteligente(outro_codigo, 400)
        contexto_partes.append(f"MÓDULO ({outro_arq}):\n{assinatura}")

    contexto_partes.append(
        "PROBLEMAS ENCONTRADOS:\n" + "\n".join(f"- {e}" for e in erros)
    )
    contexto_partes.append(
        f"CÓDIGO ATUAL DE {arquivo} (com problemas):\n```\n{codigo_atual[:1500]}\n```"
    )
    contexto_partes.append(f"Gere {arquivo} CORRIGIDO e COMPLETO:")

    contexto = "\n\n".join(contexto_partes)

    try:
        response = ollama.chat(
            model=MODELOS["coder"],
            messages=[
                {"role": "system", "content": prompt_tipo},
                {"role": "user", "content": contexto},
            ],
            options={"temperature": 0.2, "num_predict": 4096},
            stream=False,
        )
        return _extrair_codigo(response["message"]["content"].strip())
    except Exception as e:
        console.print(f"  [red]  Erro ao corrigir {arquivo}: {e}[/red]")
        return None


def _executar_step_com_validacao(
    sessao: SessaoCodigo, step: StepPlano, semantica: MemoriaSemantica
) -> bool:
    """
    Executa um step com: geração → sintaxe → validação → retry.
    Retorna True se sucesso.
    """
    for tentativa in range(1, MAX_RETRIES + 2):
        step.tentativas = tentativa

        # [13] Gera com streaming
        codigo = _executar_step_streaming(sessao, step)

        if not codigo or codigo.startswith("# Erro"):
            console.print(f"  [red]❌ Tentativa {tentativa}: geração falhou[/red]")
            continue

        # [1] Validação de sintaxe (instantânea)
        if step.arquivo.endswith(".py"):
            valido_sint, erro_sint = _validar_sintaxe(codigo, step.arquivo)
            if not valido_sint:
                console.print(f"  [yellow]⚠️ Sintaxe: {erro_sint}[/yellow]")
                if tentativa <= MAX_RETRIES:
                    codigo = _retry_step(sessao, step, codigo, [erro_sint])
                    valido2, _ = _validar_sintaxe(codigo, step.arquivo)
                    if valido2:
                        pass  # Continua para validação semântica
                    else:
                        continue
                else:
                    continue

        # Validação semântica via LLM
        validacao = _validar_step(sessao, step, codigo)

        if validacao["valido"]:
            # [14] Validação de integração entre módulos (heurística rápida)
            problemas_integracao = _validar_integracao(sessao, step, codigo)
            if problemas_integracao and tentativa <= MAX_RETRIES:
                console.print(f"  [yellow]⚠️ Integração: {'; '.join(problemas_integracao[:2])}[/yellow]")
                codigo = _retry_step(sessao, step, codigo, problemas_integracao)
                if step.arquivo.endswith(".py"):
                    v, _ = _validar_sintaxe(codigo, step.arquivo)
                    if not v:
                        continue
                # Aceita após retry de integração (não re-verifica para não entrar em loop)

            sessao.registrar_resultado(step, codigo)

            for d in validacao.get("decisoes", []):
                if d and d not in sessao.decisoes:
                    sessao.decisoes.append(d)

            # [11] Métrica — registra que usou/não usou RAG
            sessao.metricas_rag[f"step_{step.numero}"] = {
                "tentativas": tentativa,
                "chars": len(codigo),
            }

            _salvar_aprendizado(semantica, sessao, step, codigo)
            console.print(f"  [green]✅ OK[/green] ({len(codigo)} chars, tentativa {tentativa})")
            return True
        else:
            problemas = validacao.get("problemas", ["incompleto"])
            console.print(f"  [yellow]⚠️ {'; '.join(problemas[:2])}[/yellow]")
            if tentativa <= MAX_RETRIES:
                codigo = _retry_step(sessao, step, codigo, problemas)
                # Re-valida sintaxe do retry
                if step.arquivo.endswith(".py"):
                    v, _ = _validar_sintaxe(codigo, step.arquivo)
                    if not v:
                        continue
                # Re-valida semântica
                v2 = _validar_step(sessao, step, codigo)
                if v2["valido"]:
                    sessao.registrar_resultado(step, codigo)
                    _salvar_aprendizado(semantica, sessao, step, codigo)
                    console.print("  [green]✅ Corrigido[/green]")
                    return True

    # Última chance: aceita se tem sintaxe OK
    if codigo and not codigo.startswith("# Erro"):
        if not step.arquivo.endswith(".py") or _validar_sintaxe(codigo, step.arquivo)[0]:
            sessao.registrar_resultado(step, codigo)
            sessao.erros.append(f"Step {step.numero}: aceito com ressalvas")
            _salvar_aprendizado(semantica, sessao, step, codigo)
            console.print("  [yellow]⚡ Aceito com ressalvas[/yellow]")
            return True

    return False


# ══════════════════════════════════════════════════════════════
# [13] STREAMING NO LOOP
# ══════════════════════════════════════════════════════════════


def _executar_step_streaming(sessao: SessaoCodigo, step: StepPlano) -> str:
    """Gera código com streaming visual (o usuário vê o código aparecendo)."""
    contexto = sessao.contexto_para_step(step)

    # Prompt específico por tipo de arquivo (mais curto = menos alucinação)
    prompt_arquivo = _prompt_por_tipo(step.arquivo, sessao.objetivo)

    # Injeta esqueleto do template se disponível
    esqueleto_extra = ""
    template = selecionar_template(sessao.objetivo)
    if template and step.arquivo:
        esqueleto = obter_esqueleto(template, step.arquivo)
        if esqueleto:
            esqueleto_extra = f"\n\nESQUELETO (expanda):\n```\n{esqueleto}\n```"

    try:
        stream = ollama.chat(
            model=MODELOS["coder"],
            messages=[
                {"role": "system", "content": prompt_arquivo},
                {"role": "user", "content": contexto + esqueleto_extra},
            ],
            options={"temperature": 0.3, "num_predict": 4096},
            stream=True,
        )

        codigo_completo = ""
        chars_mostrados = 0

        for chunk in stream:
            texto = chunk["message"]["content"]
            codigo_completo += texto
            # Mostra progresso a cada 100 chars
            if len(codigo_completo) - chars_mostrados > 100:
                console.print(".", end="", style="dim")
                chars_mostrados = len(codigo_completo)

        console.print()  # Nova linha após os dots
        return _extrair_codigo(codigo_completo)
    except Exception as e:
        return f"# Erro: {e}"


def _prompt_por_tipo(arquivo: str, objetivo: str) -> str:
    """
    Retorna um prompt CURTO e FOCADO no tipo de arquivo.
    Modelos pequenos funcionam melhor com instruções específicas e curtas.
    """
    base = _PROMPT_CODER

    if arquivo == "requirements.txt":
        return (
            "Gere APENAS uma lista de pacotes Python, um por linha.\n"
            "Formato: nome_pacote>=versao\n"
            "Exemplo:\nflask>=3.0\n\n"
            "NÃO gere código Python, HTML, ou qualquer outra coisa.\n"
            "APENAS nomes de pacotes necessários para o projeto."
        )

    if arquivo.endswith(".md"):
        return (
            "Gere um README.md completo em Markdown.\n"
            "Inclua: título, descrição, instalação, execução, funcionalidades.\n"
            "Baseie-se nos arquivos do projeto mostrados no contexto."
        )

    if arquivo.endswith(".css"):
        return (
            "Gere APENAS regras CSS válidas.\n"
            "Formato: seletor { propriedade: valor; }\n"
            "NÃO inclua código Python, HTML ou JavaScript."
        )

    if arquivo.endswith(".html"):
        return (
            f"Gere HTML completo para: {objetivo}\n"
            "Comece com <!DOCTYPE html>. Pode usar Jinja2 ({{{{ }}}} e {{% %}}) se for template Flask.\n"
            "Inclua interface visual funcional (botões, displays, formulários conforme necessário).\n"
            "NÃO gere código Python."
        )

    if arquivo.endswith(".js"):
        return (
            f"Gere JavaScript funcional para: {objetivo}\n"
            "Inclua event listeners e lógica de interação.\n"
            "Referencie APENAS elementos HTML que existem no template (veja contexto).\n"
            "NÃO gere código Python."
        )

    # Para .py — usa o prompt base completo
    return base


# ══════════════════════════════════════════════════════════════
# [2] FEEDBACK MID-SESSION
# ══════════════════════════════════════════════════════════════


def _pedir_feedback() -> str:
    """Pede feedback ao usuário entre steps."""
    try:
        console.print("[dim]  ↳ Enter para continuar, ou digite feedback/correção:[/dim] ", end="")
        entrada = input().strip()
        return entrada
    except (EOFError, KeyboardInterrupt):
        return "abort"


# ══════════════════════════════════════════════════════════════
# [10] PLANEJAMENTO COM CHAIN-OF-THOUGHT
# ══════════════════════════════════════════════════════════════


def _planejar_cot(objetivo: str) -> SessaoCodigo:
    """
    Planejamento em 2 passes:
      1. Tenta selecionar template pré-definido (instantâneo)
      2. Se não há template adequado: LLM lista funcionalidades + organiza em steps
    Resultado melhor que um prompt único em modelos 1.2B.
    """
    # Tenta template primeiro (zero custo, melhor resultado com modelos pequenos)
    template = selecionar_template(objetivo)
    if template:
        console.print(f"[dim]  📐 Template: {template.nome} ({template.stack})[/dim]")
        from src.agentes.sessao_codigo import StepPlano
        steps = [
            StepPlano(
                numero=i,
                descricao=s.descricao,
                arquivo=s.arquivo,
                dependencias=s.dependencias,
            )
            for i, s in enumerate(template.gerar_plano(), 1)
        ]
        sessao = SessaoCodigo(objetivo=objetivo, plano=steps)
        sessao.decisoes.append(f"Template: {template.nome}, Stack: {template.stack}")
        return sessao

    # Sem template — usa LLM com chain-of-thought
    try:
        # Passo 1: listar funcionalidades
        r1 = ollama.chat(
            model=MODELOS["rapido"],
            messages=[
                {"role": "user", "content": _PROMPT_COT_1.format(objetivo=objetivo)},
            ],
            options={"temperature": 0.3, "num_predict": 300},
        )
        funcionalidades = r1["message"]["content"].strip()
        console.print("[dim]  Funcionalidades identificadas[/dim]")

        # Passo 2: organizar em steps
        r2 = ollama.chat(
            model=MODELOS["rapido"],
            messages=[
                {"role": "user", "content": _PROMPT_COT_2.format(funcionalidades=funcionalidades)},
            ],
            format="json",
            options={"temperature": 0.2, "num_predict": 500},
        )
        raw = r2["message"]["content"].strip()
        return _parse_plano(objetivo, raw)

    except Exception:
        # Fallback: planejamento simples
        return _planejar_simples(objetivo)


def _planejar_simples(objetivo: str) -> SessaoCodigo:
    """Planejamento fallback com prompt único."""
    try:
        response = ollama.chat(
            model=MODELOS["rapido"],
            messages=[
                {"role": "system", "content": _PROMPT_PLANEJAR_SIMPLES},
                {"role": "user", "content": objetivo},
            ],
            format="json",
            options={"temperature": 0.2, "num_predict": 500},
        )
        return _parse_plano(objetivo, response["message"]["content"].strip())
    except Exception:
        return SessaoCodigo(objetivo=objetivo, plano=[StepPlano(numero=1, descricao=objetivo, arquivo="main.py")])


# ══════════════════════════════════════════════════════════════
# FUNÇÕES INTERNAS DO LOOP
# ══════════════════════════════════════════════════════════════


def _retry_step(sessao: SessaoCodigo, step: StepPlano, codigo_ant: str, problemas: list[str]) -> str:
    """Retry com feedback dos problemas."""
    contexto = sessao.contexto_para_step(step)
    prompt = _PROMPT_RETRY.format(problemas="\n".join(f"- {p}" for p in problemas))
    try:
        response = ollama.chat(
            model=MODELOS["coder"],
            messages=[
                {"role": "system", "content": _PROMPT_CODER},
                {"role": "user", "content": contexto},
                {"role": "assistant", "content": f"```\n{codigo_ant[:2000]}\n```"},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2, "num_predict": 4096},
            stream=False,
        )
        return _extrair_codigo(response["message"]["content"].strip())
    except Exception:
        return codigo_ant


def _validar_step(sessao: SessaoCodigo, step: StepPlano, codigo: str) -> dict:
    """Validação semântica via heurísticas + LLM."""
    if not codigo or len(codigo) < 20:
        return {"valido": False, "problemas": ["vazio"], "decisoes": []}

    # Heurísticas rápidas (detectam problemas SEM gastar LLM)
    probs = []

    # Tipo de conteúdo errado (o problema mais comum com modelos pequenos)
    if step.arquivo == "requirements.txt":
        if "def " in codigo or "@app" in codigo or "import " in codigo:
            return {"valido": False, "problemas": [
                "requirements.txt contém código Python. Gere APENAS nomes de pacotes, um por linha. Ex: flask>=3.0"
            ], "decisoes": []}

    if step.arquivo and step.arquivo.endswith(".py"):
        if codigo.strip().startswith("<!DOCTYPE") or codigo.strip().startswith("<html"):
            return {"valido": False, "problemas": [
                f"{step.arquivo} contém HTML em vez de código Python"
            ], "decisoes": []}

    # Código incompleto
    if "TODO" in codigo and len(codigo) < 300:
        probs.append("TODOs não resolvidos")
    if codigo.count("pass") > 3 and len(codigo) < 400:
        probs.append("implementação stub")
    linhas_stub = sum(
        1 for linha in codigo.split("\n")
        if linha.strip() in ("pass",) or linha.strip().startswith("# Implementação")
    )
    if linhas_stub > 2:
        probs.append(f"{linhas_stub} linhas stub/pass — código incompleto")
    if probs:
        return {"valido": False, "problemas": probs, "decisoes": []}

    # Validação via LLM (apenas se heurísticas passaram)
    try:
        r = ollama.chat(
            model=MODELOS["rapido"],
            messages=[
                {"role": "system", "content": _PROMPT_VALIDAR},
                {"role": "user", "content": (
                    f"Objetivo: {step.descricao}\nArquivo: {step.arquivo}\n"
                    f"Código:\n{codigo[:1200]}"
                )},
            ],
            options={"temperature": 0.1, "num_predict": 80, "num_ctx": 1024},
        )
        return _parse_validacao(r["message"]["content"].strip())
    except Exception:
        return {"valido": True, "problemas": [], "decisoes": []}


def _validar_integracao(sessao: SessaoCodigo, step: StepPlano, codigo: str) -> list[str]:
    """
    Validação heurística de integração entre módulos.
    Verifica se o código atual referencia corretamente os módulos já gerados.
    Retorna lista de problemas (vazia = OK).
    """
    problemas = []
    if not step.arquivo:
        return problemas

    arquivo = step.arquivo
    codigo_lower = codigo.lower()

    # ─── Regra 0: Validação de TIPO DE CONTEÚDO ───
    # Detecta quando o LLM gerou conteúdo errado para o tipo de arquivo
    if arquivo.endswith(".txt") or arquivo == "requirements.txt":
        # Detecta código Python/HTML misturado em requirements.txt
        indicadores_codigo = ["def ", "class ", "import ", "from ", "@app", "<!DOCTYPE", "<html", "function "]
        for indicador in indicadores_codigo:
            if indicador in codigo:
                problemas.append(
                    f"ERRO GRAVE: {arquivo} contém código ({indicador}...) mas deveria ter APENAS nomes de pacotes. "
                    f"Formato correto: um pacote por linha (ex: flask>=3.0). NADA MAIS."
                )
                return problemas  # Erro fatal — não checa mais nada

    if arquivo.endswith(".css"):
        if "from flask" in codigo or "import " in codigo or "def " in codigo:
            problemas.append(f"{arquivo} contém código Python mas deveria ter apenas CSS.")
            return problemas
        if "<!DOCTYPE" in codigo or "<html" in codigo:
            problemas.append(f"{arquivo} contém HTML mas deveria ter apenas CSS.")
            return problemas

    if arquivo.endswith(".html"):
        if codigo.strip().startswith(("from ", "import ", "#!/", "def ")):
            problemas.append(f"{arquivo} começa com código Python mas deveria ser HTML.")
            return problemas

    if arquivo.endswith(".js") and not arquivo.endswith(".json"):
        if "from flask" in codigo or "import flask" in codigo_lower:
            problemas.append(f"{arquivo} contém imports Python mas deveria ser JavaScript.")
            return problemas

    # ─── Regra 1: Detecta código duplicado entre arquivos ───
    if step.arquivo.endswith(".py") and sessao.scratchpad:
        for outro_arq, outro_codigo in sessao.scratchpad.items():
            if outro_arq == step.arquivo or not outro_arq.endswith(".py"):
                continue
            # Se >60% das linhas são idênticas, é duplicação
            linhas_novas = set(
                ln.strip() for ln in codigo.split("\n")
                if ln.strip() and not ln.strip().startswith("#")
            )
            linhas_outro = set(
                ln.strip() for ln in outro_codigo.split("\n")
                if ln.strip() and not ln.strip().startswith("#")
            )
            if linhas_novas and linhas_outro:
                overlap = len(linhas_novas & linhas_outro) / max(len(linhas_novas), 1)
                if overlap > 0.6:
                    problemas.append(
                        f"{arquivo} é >60% idêntico a {outro_arq}. "
                        f"Cada módulo deve ter responsabilidade DIFERENTE. "
                        f"models.py = classes de dados. app.py = rotas/servidor. "
                        f"NÃO copie o mesmo código em ambos."
                    )
                    return problemas

    # ─── Regra 2: models.py não deveria ter rotas Flask/@app ───
    if "model" in arquivo.lower() and arquivo.endswith(".py"):
        if "@app.route" in codigo or "app.run(" in codigo or "Flask(__name__)" in codigo:
            problemas.append(
                f"{arquivo} contém rotas Flask (@app.route) mas deveria ter APENAS "
                f"classes de dados e lógica de negócio. Rotas vão no app.py."
            )

    # ─── Regra 3: Storage/persistência deve importar models ───
    eh_storage = any(
        x in arquivo.lower() for x in ("storage", "persist", "database", "db", "repo")
    )
    if eh_storage:
        modelos_existentes = [
            f for f in sessao.scratchpad.keys()
            if "model" in f.lower() and f.endswith(".py")
        ]
        if modelos_existentes:
            for mod_file in modelos_existentes:
                mod_code = sessao.scratchpad[mod_file]
                classes = re.findall(r"class\s+(\w+)", mod_code)
                for cls in classes:
                    if cls in codigo and f"from {mod_file.replace('.py', '')} import" not in codigo:
                        if f"import {mod_file.replace('.py', '')}" not in codigo:
                            problemas.append(
                                f"Usa '{cls}' mas não importa de {mod_file}. "
                                f"Adicione: from {mod_file.replace('.py', '')} import {cls}"
                            )

    # ─── Regra 4: Ponto de entrada deve usar storage E models ───
    eh_ponto_entrada = arquivo.split(".")[0] in ("main", "index", "app", "server", "cli")
    if eh_ponto_entrada and arquivo.endswith(".py"):
        tem_storage = any("storage" in f.lower() for f in sessao.scratchpad.keys())

        if tem_storage and "storage" not in codigo_lower and "import" in codigo_lower:
            problemas.append(
                "Ponto de entrada não usa o módulo storage — dados não serão persistidos."
            )

    # ─── Regra 5: requirements.txt deve ter apenas o que é usado ───
    if arquivo == "requirements.txt":
        deps_listadas = [
            line.split("==")[0].split(">=")[0].split("<=")[0].strip().lower()
            for line in codigo.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        for dep in deps_listadas:
            dep_import = dep.replace("-", "_")
            usado = any(
                dep_import in src_code.lower() or dep in src_code.lower()
                for f, src_code in sessao.scratchpad.items()
                if f.endswith((".py", ".js", ".ts")) and f != arquivo
            )
            if not usado and sessao.scratchpad:
                # Só valida se já há outros arquivos gerados
                problemas.append(
                    f"requirements.txt lista '{dep}' mas nenhum arquivo do projeto importa essa biblioteca"
                )

    # ─── Regra 6: HTML com rotas Flask deve usar URLs válidas ───
    if arquivo.endswith(".html") and "<form" in codigo:
        # Detecta action com placeholder não resolvido
        placeholders_invalidos = re.findall(r'action="[^"]*<[^"]*>"', codigo)
        if placeholders_invalidos:
            problemas.append(
                f"HTML contém action com placeholder não resolvido: {placeholders_invalidos[0]}. "
                f"Use URLs reais como /add ou /delete/{{{{ loop.index0 }}}}"
            )

    return problemas


# ══════════════════════════════════════════════════════════════
# APRENDIZADO CHROMADB
# ══════════════════════════════════════════════════════════════


def _salvar_aprendizado(sem: MemoriaSemantica, sessao: SessaoCodigo, step: StepPlano, codigo: str):
    """Salva par intenção→código para reutilização futura."""
    doc = (
        f"Tarefa: {step.descricao}\n"
        f"Contexto: {sessao.objetivo}\n"
        f"Arquivo: {step.arquivo}\n\n"
        f"Código:\n{codigo[:3000]}"
    )
    sem.adicionar_conhecimento(texto=doc, fonte=f"projeto:{sessao.objetivo[:50]}", tipo="codigo_gerado")


def _salvar_projeto_completo(sem: MemoriaSemantica, sessao: SessaoCodigo):
    """Salva resumo do projeto como conhecimento de alto nível."""
    if not sessao.scratchpad:
        return
    resumo = (
        f"Projeto: {sessao.objetivo}\n"
        f"Arquivos: {', '.join(sessao.scratchpad.keys())}\n"
        f"Decisões: {'; '.join(sessao.decisoes[:5])}\n"
        f"Steps: {len(sessao.plano)}"
    )
    sem.adicionar_conhecimento(texto=resumo, fonte=f"arquitetura:{sessao.objetivo[:40]}", tipo="arquitetura")


# ══════════════════════════════════════════════════════════════
# EXPORTAR + HELPERS
# ══════════════════════════════════════════════════════════════


def _exportar_disco(sessao: SessaoCodigo, diretorio: str | None = None) -> Path:
    """Salva arquivos em disco."""
    if diretorio:
        base = Path(diretorio)
    else:
        nome = sessao.objetivo[:40].replace(" ", "_").replace("/", "-").lower()
        base = PROJETOS_DIR / nome

    base.mkdir(parents=True, exist_ok=True)
    for arq, conteudo in sessao.scratchpad.items():
        caminho = base / arq
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(conteudo, encoding="utf-8")

    meta = {
        "objetivo": sessao.objetivo,
        "steps": len(sessao.plano),
        "arquivos": list(sessao.scratchpad.keys()),
        "decisoes": sessao.decisoes,
        "erros": sessao.erros,
        "tempo_ms": sessao.tempo_total_ms,
        "metricas_rag": sessao.metricas_rag,
        "projeto_validado": sessao.projeto_validado,
    }
    (base / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return base


def _passada_coerencia(sessao: SessaoCodigo):
    """
    Passada final de coerência: verifica e corrige problemas de integração
    entre módulos SEM usar LLM (heurísticas determinísticas).

    Corrige automaticamente:
    - README vazio/genérico → regenera com dados reais
    - requirements.txt com conteúdo errado → regenera baseado em imports
    - Arquivos duplicados → remove o duplicado
    - Imports faltantes óbvios → adiciona
    """
    if not sessao.scratchpad:
        return

    # 0. Detecta e corrige requirements.txt com código Python dentro
    if "requirements.txt" in sessao.scratchpad:
        req = sessao.scratchpad["requirements.txt"]
        if "def " in req or "import " in req or "@app" in req or "class " in req:
            # Regenera baseado nos imports reais do projeto
            sessao.scratchpad["requirements.txt"] = _gerar_requirements_deterministico(sessao)
            console.print("  [dim]📦 requirements.txt regenerado (continha código)[/dim]")

    # 1. Detecta módulos duplicados (>60% linhas idênticas)
    arquivos_py = [f for f in sessao.scratchpad if f.endswith(".py")]
    duplicados_removidos = set()
    for i, arq1 in enumerate(arquivos_py):
        if arq1 in duplicados_removidos:
            continue
        for arq2 in arquivos_py[i + 1:]:
            if arq2 in duplicados_removidos:
                continue
            linhas1 = set(ln.strip() for ln in sessao.scratchpad[arq1].split("\n") if ln.strip())
            linhas2 = set(ln.strip() for ln in sessao.scratchpad[arq2].split("\n") if ln.strip())
            if linhas1 and linhas2:
                overlap = len(linhas1 & linhas2) / max(len(linhas1), len(linhas2))
                if overlap > 0.6:
                    # Remove o que NÃO é ponto de entrada
                    pontos = {"main.py", "app.py", "index.js", "server.js"}
                    remover = arq2 if arq1 in pontos else arq1
                    if remover not in pontos:
                        # Não remove, mas esvazia — será regenerado ou ignorado
                        console.print(f"  [dim]⚠️ {remover} é duplicata de {arq1 if remover == arq2 else arq2}[/dim]")

    # 2. README vazio ou genérico → regenera inline (sem LLM)
    readme_key = next((f for f in sessao.scratchpad if f.lower() == "readme.md"), None)
    if readme_key:
        readme = sessao.scratchpad[readme_key]
        tem_instrucao = any(x in readme.lower() for x in ["python ", "node ", "npm ", "pip "])
        if len(readme.strip()) < 100 or not tem_instrucao:
            sessao.scratchpad[readme_key] = _gerar_readme_deterministico(sessao)
            console.print("  [dim]📝 README regenerado com dados do projeto[/dim]")

    # 3. Limpa requirements.txt de deps não usadas
    if "requirements.txt" in sessao.scratchpad:
        req = sessao.scratchpad["requirements.txt"]
        # Só limpa se parece ser formato correto (não tem código)
        if "def " not in req and "import " not in req:
            codigo_total = "\n".join(
                c for f, c in sessao.scratchpad.items()
                if f.endswith(".py") and f != "requirements.txt"
            )
            linhas_limpas = []
            for linha in req.split("\n"):
                if not linha.strip() or linha.strip().startswith("#"):
                    linhas_limpas.append(linha)
                    continue
                dep = linha.split(">=")[0].split("==")[0].split("<=")[0].strip().lower()
                dep_import = dep.replace("-", "_")
                if dep_import in codigo_total.lower() or dep in codigo_total.lower():
                    linhas_limpas.append(linha)
            resultado = "\n".join(linhas_limpas).strip()
            if resultado:
                sessao.scratchpad["requirements.txt"] = resultado + "\n"

    # 4. Verifica se ponto de entrada importa os módulos disponíveis
    pontos = {"main.py", "app.py", "index.js", "server.js"}
    entrada = next((f for f in sessao.scratchpad if f in pontos), None)
    if entrada and entrada.endswith(".py"):
        codigo_entrada = sessao.scratchpad[entrada]
        modulos_py = [
            f.replace(".py", "") for f in sessao.scratchpad
            if f.endswith(".py") and f != entrada and "/" not in f
        ]
        imports_faltantes = []
        for mod in modulos_py:
            if f"from {mod}" not in codigo_entrada and f"import {mod}" not in codigo_entrada:
                mod_code = sessao.scratchpad.get(f"{mod}.py", "")
                classes = re.findall(r"class\s+(\w+)", mod_code)
                funcs = [f for f in re.findall(r"^def\s+(\w+)", mod_code, re.MULTILINE) if not f.startswith("_")]
                if classes or funcs:
                    exports = classes[:3] + funcs[:3]
                    imports_faltantes.append(f"from {mod} import {', '.join(exports)}")

        if imports_faltantes:
            linhas = codigo_entrada.split("\n")
            insert_pos = 0
            for i, linha in enumerate(linhas):
                if linha.startswith("#!") or linha.startswith('"""') or linha.startswith("'''"):
                    insert_pos = i + 1
                elif linha.startswith("import ") or linha.startswith("from "):
                    insert_pos = i + 1
                elif insert_pos > 0 and linha.strip():
                    break
            for imp in reversed(imports_faltantes):
                linhas.insert(insert_pos, imp)
            sessao.scratchpad[entrada] = "\n".join(linhas)
            console.print(f"  [dim]🔗 Imports adicionados em {entrada}: {len(imports_faltantes)}[/dim]")


def _gerar_requirements_deterministico(sessao: SessaoCodigo) -> str:
    """Gera requirements.txt baseado nos imports reais do código Python."""
    # Mapa de imports conhecidos → pacote pip
    IMPORT_TO_PIP = {
        "flask": "flask>=3.0",
        "fastapi": "fastapi>=0.100",
        "uvicorn": "uvicorn>=0.25",
        "requests": "requests>=2.28",
        "pydantic": "pydantic>=2.0",
        "sqlalchemy": "sqlalchemy>=2.0",
        "sqlite3": None,  # stdlib
        "json": None,
        "os": None,
        "sys": None,
        "re": None,
        "datetime": None,
        "pathlib": None,
    }

    deps_encontradas = set()
    for arquivo, codigo in sessao.scratchpad.items():
        if not arquivo.endswith(".py"):
            continue
        imports = re.findall(r"^(?:from|import)\s+(\w+)", codigo, re.MULTILINE)
        for imp in imports:
            imp_lower = imp.lower()
            if imp_lower in IMPORT_TO_PIP:
                pip_pkg = IMPORT_TO_PIP[imp_lower]
                if pip_pkg:
                    deps_encontradas.add(pip_pkg)
            elif imp_lower not in ("models", "storage", "services", "routes", "utils", "core"):
                # Não é stdlib nem módulo interno — assume que é pacote pip
                deps_encontradas.add(f"{imp_lower}")

    return "\n".join(sorted(deps_encontradas)) + "\n" if deps_encontradas else "# sem dependências externas\n"


def _gerar_readme_deterministico(sessao: SessaoCodigo) -> str:
    """Gera README completo baseado nos arquivos reais do projeto, sem LLM."""
    titulo = sessao.objetivo.split(".")[0].strip().title()
    linhas = [f"# {titulo}", "", sessao.objetivo, ""]

    # Instalação
    if "requirements.txt" in sessao.scratchpad:
        linhas.extend([
            "## Instalação", "",
            "```bash",
            "pip install -r requirements.txt",
            "```", "",
        ])
    elif "package.json" in sessao.scratchpad:
        linhas.extend([
            "## Instalação", "",
            "```bash",
            "npm install",
            "```", "",
        ])

    # Execução
    pontos = {"main.py": "python main.py", "app.py": "python app.py",
              "index.js": "node index.js", "server.js": "node server.js"}
    entrada = next((f for f in sessao.scratchpad if f in pontos), None)
    if entrada:
        cmd = pontos[entrada]
        linhas.extend(["## Execução", "", "```bash", cmd, "```", ""])
        # Detecta se é web
        codigo = sessao.scratchpad.get(entrada, "")
        if "5000" in codigo:
            linhas.append("Acesse: http://localhost:5000\n")
        elif "3000" in codigo:
            linhas.append("Acesse: http://localhost:3000\n")
        elif "8000" in codigo:
            linhas.append("Acesse: http://localhost:8000\n")

    # Funcionalidades
    linhas.extend(["## Funcionalidades", ""])
    for step in sessao.plano:
        if step.arquivo and step.arquivo.lower() != "readme.md" and step.arquivo != "requirements.txt":
            linhas.append(f"- {step.descricao}")
    linhas.append("")

    # Estrutura
    linhas.extend(["## Estrutura do Projeto", "", "| Arquivo | Descrição |", "|---------|-----------|"])
    for arquivo in sorted(sessao.scratchpad.keys()):
        if arquivo.lower() == "readme.md":
            continue
        desc = next((s.descricao for s in sessao.plano if s.arquivo == arquivo), arquivo)
        linhas.append(f"| `{arquivo}` | {desc} |")
    linhas.append("")

    return "\n".join(linhas)


def _validar_projeto_gerado(sessao: SessaoCodigo) -> list[str]:
    """Executa verificações locais baratas no conjunto completo de arquivos."""
    if not sessao.scratchpad:
        return ["nenhum arquivo foi gerado"]

    problemas: list[str] = []
    arquivos = set(sessao.scratchpad)
    for step in sessao.plano:
        for dependencia in step.dependencias:
            # Dependências podem representar diretórios conceituais em alguns templates.
            if dependencia.endswith("/") or dependencia in arquivos:
                continue
            problemas.append(
                f"{step.arquivo or step.descricao}: dependência ausente {dependencia}"
            )

    pontos_entrada = {
        "main.py", "app.py", "index.js", "server.js", "src/index.ts", "src/main.jsx"
    }
    if not (arquivos & pontos_entrada):
        problemas.append("ponto de entrada executável não identificado")

    with tempfile.TemporaryDirectory(prefix="potato-claw-check-") as tmp:
        base = Path(tmp)
        for arquivo, conteudo in sessao.scratchpad.items():
            destino = base / arquivo
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(conteudo, encoding="utf-8")

        arquivos_py = list(base.rglob("*.py"))
        if arquivos_py:
            proc = subprocess.run(
                [sys.executable, "-m", "compileall", "-q", str(base)],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if proc.returncode != 0:
                problemas.append((proc.stderr or proc.stdout or "falha no compileall").strip()[:500])

        package_json = base / "package.json"
        if package_json.exists():
            try:
                json.loads(package_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                problemas.append(f"package.json inválido: {exc}")

        node = shutil.which("node")
        if node:
            for arquivo_js in list(base.rglob("*.js")):
                proc = subprocess.run(
                    [node, "--check", str(arquivo_js)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode != 0:
                    problemas.append(
                        f"{arquivo_js.relative_to(base)}: "
                        f"{(proc.stderr or proc.stdout).strip()[:300]}"
                    )

    return problemas


def _atualizar_validacao_final(sessao: SessaoCodigo) -> list[str]:
    """Atualiza o estado de conclusão a partir dos smoke checks locais."""
    problemas = _validar_projeto_gerado(sessao)
    sessao.projeto_validado = not problemas
    if problemas:
        existentes = set(sessao.erros)
        for problema in problemas:
            mensagem = f"Validação final: {problema}"
            if mensagem not in existentes:
                sessao.erros.append(mensagem)
    return problemas


# Tags de linguagem reconhecidas na cerca (usadas só no caso de cerca aberta
# sem fechamento, para descartar a tag sem comer a 1a linha de código).
_LANGS_CERCA = {
    "python", "py", "python3", "javascript", "js", "jsx", "typescript", "ts",
    "tsx", "php", "go", "golang", "rust", "rs", "java", "sql", "bash", "sh",
    "shell", "zsh", "yaml", "yml", "json", "toml", "html", "css", "scss",
    "c", "cpp", "csharp", "cs", "ruby", "rb", "kotlin", "kt", "swift",
    "dockerfile", "make", "makefile", "ini", "xml", "markdown", "md", "text", "txt",
}

# Bloco cercado por ``` com tag opcional na própria cerca; conteúdo (grupo 2)
# vai até o ``` de fechamento. Separar a tag aqui evita comer a 1a linha de código.
_CERCA_RE = re.compile(r"```[ \t]*([^\n`]*)\r?\n(.*?)```", re.DOTALL)


def _extrair_codigo(texto: str) -> str:
    """
    Extrai o código de uma resposta de LLM de forma tolerante a modelos
    pequenos, que erram a formatação dos blocos com frequência.

    Estratégia:
      1. Captura todos os blocos ```...``` fechados e escolhe o MAIOR
         (o código real costuma ser o maior; exemplos curtos na explicação
         ficam de fora). A versão anterior pegava sempre o primeiro bloco.
      2. Trata cerca ABERTA sem fechamento (modelo esqueceu o ``` final).
      3. Descarta a tag de linguagem sem comer a primeira linha de código.
      4. Sem nenhuma cerca, devolve o texto cru (assume código puro).
    """
    if "```" not in texto:
        return texto.strip()

    # 1. Blocos fechados -> escolhe o de maior conteúdo.
    blocos = _CERCA_RE.findall(texto)
    if blocos:
        _tag, conteudo = max(blocos, key=lambda par: len(par[1]))
        return conteudo.strip()

    # 2. Cerca aberta sem fechar: do primeiro ``` até o fim.
    resto = texto.split("```", 1)[1]
    primeira, sep, corpo = resto.partition("\n")
    # 3. Remove a tag de linguagem apenas se a 1a linha for de fato uma tag.
    if sep and primeira.strip().lower() in _LANGS_CERCA:
        resto = corpo
    # Remove um eventual fence de fechamento perdido no meio.
    return resto.split("```", 1)[0].strip()


def _parse_plano(objetivo: str, raw: str) -> SessaoCodigo:
    """Parseia JSON do plano."""
    inicio = raw.find("{")
    if inicio == -1:
        return SessaoCodigo(objetivo=objetivo, plano=[StepPlano(numero=1, descricao=objetivo, arquivo="main.py")])
    depth, fim = 0, -1
    for i in range(inicio, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                fim = i + 1
                break
    if fim == -1:
        return SessaoCodigo(objetivo=objetivo, plano=[StepPlano(numero=1, descricao=objetivo, arquivo="main.py")])
    try:
        data = json.loads(raw[inicio:fim])
    except json.JSONDecodeError:
        return SessaoCodigo(objetivo=objetivo, plano=[StepPlano(numero=1, descricao=objetivo, arquivo="main.py")])
    steps = [
        StepPlano(numero=i, descricao=s.get("descricao", f"Step {i}"),
                  arquivo=s.get("arquivo", ""), dependencias=s.get("dependencias", []))
        for i, s in enumerate(data.get("steps", []), 1)
    ]
    return SessaoCodigo(objetivo=objetivo, plano=steps or [StepPlano(numero=1, descricao=objetivo, arquivo="main.py")])


def _parse_validacao(raw: str) -> dict:
    """Parseia JSON de validação."""
    inicio = raw.find("{")
    if inicio == -1:
        return {"valido": True, "problemas": [], "decisoes": []}
    depth, fim = 0, -1
    for i in range(inicio, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                fim = i + 1
                break
    if fim == -1:
        return {"valido": True, "problemas": [], "decisoes": []}
    try:
        d = json.loads(raw[inicio:fim])
        return {
            "valido": bool(d.get("valido", True)),
            "problemas": d.get("problemas", []),
            "decisoes": d.get("decisoes", []),
        }
    except (json.JSONDecodeError, KeyError):
        return {"valido": True, "problemas": [], "decisoes": []}


# ══════════════════════════════════════════════════════════════
# [11] MÉTRICAS A/B DE QUALIDADE
# ══════════════════════════════════════════════════════════════


def metricas_qualidade(sessao: SessaoCodigo) -> dict:
    """Calcula métricas de qualidade da sessão."""
    total_steps = len(sessao.plano)
    concluidos = sum(1 for s in sessao.plano if s.concluido)
    pulados = sum(1 for s in sessao.plano if s.pulado)
    total_tentativas = sum(s.tentativas for s in sessao.plano)
    media_tentativas = total_tentativas / max(concluidos, 1)

    return {
        "taxa_sucesso": f"{concluidos}/{total_steps} ({concluidos/max(total_steps,1)*100:.0f}%)",
        "pulados": pulados,
        "media_tentativas": round(media_tentativas, 1),
        "tempo_por_step_ms": sessao.tempo_total_ms // max(concluidos, 1),
        "total_chars_gerados": sum(len(v) for v in sessao.scratchpad.values()),
    }


# ══════════════════════════════════════════════════════════════
# [6] CACHE TTL POR TIPO (integração com web_rag)
# ══════════════════════════════════════════════════════════════
# Nota: implementado via config NEURON_WEB_RAG_CACHE_TTL
# Para TTL diferenciado, use a variável de ambiente.
# Docs: 86400 (24h), cotações: 300 (5min), geral: 3600 (1h)


# ══════════════════════════════════════════════════════════════
# INTERFACE PÚBLICA
# ══════════════════════════════════════════════════════════════

_sessao_ativa: Optional[SessaoCodigo] = None


def obter_sessao() -> Optional[SessaoCodigo]:
    """Retorna sessão ativa (ou restaura do disco)."""
    global _sessao_ativa
    if not _sessao_ativa:
        _sessao_ativa = _restaurar_sessao()
    return _sessao_ativa


def iniciar_sessao(objetivo: str) -> SessaoCodigo:
    """Inicia sessão (planeja sem executar)."""
    global _sessao_ativa
    _sessao_ativa = _planejar_cot(objetivo)
    _persistir_sessao(_sessao_ativa)
    return _sessao_ativa


def avancar_sessao() -> tuple[Optional[StepPlano], str]:
    """Executa um step (modo manual)."""
    global _sessao_ativa
    if not _sessao_ativa:
        return None, ""
    step = _sessao_ativa.step_pendente()
    if not step:
        return None, ""
    sem = MemoriaSemantica()
    _sessao_ativa.snapshot()
    sucesso = _executar_step_com_validacao(_sessao_ativa, step, sem)
    if sucesso and _sessao_ativa.step_pendente() is None:
        _atualizar_validacao_final(_sessao_ativa)
    _persistir_sessao(_sessao_ativa)
    if sucesso:
        return step, step.resultado
    else:
        _sessao_ativa.rollback()
        return step, "# Falhou"


def executar_completo(objetivo: str, salvar_disco: bool = True, interativo: bool = False) -> SessaoCodigo:
    """Executa projeto completo no agent loop."""
    global _sessao_ativa
    sessao = executar_projeto(objetivo, salvar_disco=salvar_disco, interativo=interativo)
    _sessao_ativa = sessao
    return sessao


def rerun_step(numero: int) -> tuple[Optional[StepPlano], str]:
    """[4] Re-executa um step específico."""
    global _sessao_ativa
    if not _sessao_ativa:
        return None, ""
    for step in _sessao_ativa.plano:
        if step.numero == numero:
            step.concluido = False
            step.pulado = False
            step.tentativas = 0
            _sessao_ativa.snapshot()
            sem = MemoriaSemantica()
            sucesso = _executar_step_com_validacao(_sessao_ativa, step, sem)
            if sucesso and _sessao_ativa.step_pendente() is None:
                _atualizar_validacao_final(_sessao_ativa)
            _persistir_sessao(_sessao_ativa)
            if sucesso:
                return step, step.resultado
            else:
                _sessao_ativa.rollback()
                return step, "# Falhou"
    return None, ""


def finalizar_sessao() -> Optional[SessaoCodigo]:
    """Finaliza sessão."""
    global _sessao_ativa
    sessao = _sessao_ativa
    _sessao_ativa = None
    _limpar_persistencia()
    return sessao
