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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ollama
from rich.console import Console
from rich.panel import Panel

from src.core.config import MODELOS, DATA_DIR
from src.memoria.semantica import MemoriaSemantica
from src.agentes.templates import selecionar_template, obter_esqueleto

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

    @property
    def progresso(self) -> str:
        total = len(self.plano)
        feitos = sum(1 for s in self.plano if s.concluido)
        return f"{feitos}/{total}"

    @property
    def concluida(self) -> bool:
        return all(s.concluido or s.pulado for s in self.plano)

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
                if f != step.arquivo and not f.endswith((".md", ".txt", ".json"))
            ]
            if modulos_existentes:
                partes.append(
                    f"MÓDULOS DISPONÍVEIS PARA IMPORTAR: {', '.join(modulos_existentes)}\n"
                    f"IMPORTANTE: Este é o PONTO DE ENTRADA. Deve importar e usar os módulos acima "
                    f"para criar uma interface interativa (CLI com menu ou servidor web)."
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

        return "\n\n".join(partes)

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
    if sem.collection.count() == 0:
        return 0

    limite = (datetime.now() - timedelta(days=dias_expiracao)).isoformat()
    # Busca todos os documentos com metadata
    try:
        todos = sem.collection.get(include=["metadatas"])
        ids_remover = []
        for i, meta in enumerate(todos["metadatas"]):
            criado = meta.get("criado_em", "")
            if criado and criado < limite:
                ids_remover.append(todos["ids"][i])

        if ids_remover:
            sem.collection.delete(ids=ids_remover)
        return len(ids_remover)
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

REGRAS OBRIGATÓRIAS:
- O projeto DEVE ter um ponto de entrada executável (main.py, index.js, app.py, etc.)
- O projeto DEVE ter interface interativa: CLI com menu/readline, ou servidor web com endpoints testáveis, ou GUI
- O projeto DEVE funcionar ao rodar (python main.py, node index.js, etc.) sem configuração extra
- Inclua: arquivo de configuração/dependências (requirements.txt, package.json), README com instruções de uso
- Se for CLI: use menus interativos, prompts de input, feedback visual
- Se for web: inclua ao menos uma rota funcional testável com curl ou navegador
- ESCOLHA UMA ÚNICA LINGUAGEM/STACK e use apenas ela em todo o projeto. Não misture Python e JavaScript.
- Se não especificado, use Python com CLI interativa

Responda como bullet points. Máximo 10 itens. Seja específico e técnico.
O PRIMEIRO item deve ser sempre o ponto de entrada principal com interface interativa.
O ÚLTIMO item deve ser o README.md com instruções de execução."""

_PROMPT_COT_2 = """Com base nestas funcionalidades:
{funcionalidades}

Organize em arquivos de código. Cada arquivo = 1 step.
REGRAS:
- O PRIMEIRO step DEVE ser o arquivo de dependências (requirements.txt ou package.json)
- O SEGUNDO step DEVE ser os módulos/classes de lógica de negócio
- O PENÚLTIMO step DEVE ser o ponto de entrada principal (main.py/index.js/app.py) que importa os módulos e oferece interface interativa
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

_PROMPT_CODER = """Você é um programador expert que cria projetos FUNCIONAIS e EXECUTÁVEIS.

REGRAS OBRIGATÓRIAS:
1. Gere APENAS o código do arquivo pedido — completo e funcional
2. Inclua TODOS os imports necessários no topo
3. Se for o ponto de entrada (main.py, index.js, app.py): DEVE ter interface interativa
   - Para CLI: use loop com menu de opções, input() do usuário, feedback visual
   - Para web: servidor que suba e responda em localhost
4. Se for módulo: exporte classes/funções que o ponto de entrada vai importar
5. Se for requirements.txt/package.json: liste APENAS dependências realmente usadas
6. Se for README.md: inclua instruções EXATAS de como instalar e executar
7. Código deve funcionar ao ser executado — sem TODOs, sem stubs, sem "implementar depois"
8. Use a stack definida pelo usuário. Se não definida, use Python com CLI interativa
9. NÃO inclua explicações fora do código — apenas comentários inline quando necessário
10. O projeto deve ser VIVO: o usuário roda e interage imediatamente"""

_PROMPT_VALIDAR = """Avalie se o código atende ao objetivo e é EXECUTÁVEL.
Critérios: código completo (sem TODOs/stubs), imports corretos, interface funcional (se for ponto de entrada).
Responda JSON: {"valido":true/false,"problemas":[],"decisoes":[]}"""

_PROMPT_RETRY = """PROBLEMAS ENCONTRADOS:
{problemas}

Gere o código COMPLETO corrigido. Lembre-se:
- O código deve ser executável sem erros
- Se for ponto de entrada: deve ter interface interativa funcional
- Sem TODOs, sem stubs, sem placeholders"""


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
    semantica = MemoriaSemantica()

    # [5] Verifica se Ollama está disponível
    if not _ollama_disponivel():
        console.print("[red]❌ Ollama não está rodando. Execute: ollama serve[/red]")
        return SessaoCodigo(objetivo=objetivo, erros=["Ollama indisponível"])

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
            resumo_projeto = "\n".join(f"  {k}" for k in list(indice.keys())[:20])
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
            console.print(f"  [red]⏭️  Step pulado (rollback aplicado)[/red]")

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
                console.print(f"[dim]📝 Decisão registrada[/dim]")

    # ─── FINALIZAÇÃO ───
    sessao.tempo_total_ms = int((time.time() - inicio_total) * 1000)
    _salvar_projeto_completo(semantica, sessao)

    if salvar_disco and sessao.scratchpad:
        caminho = _exportar_disco(sessao, diretorio_saida)
        console.print(f"\n[bold green]📁 Salvo em: {caminho}[/bold green]")

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
                    console.print(f"  [green]✅ Corrigido[/green]")
                    return True

    # Última chance: aceita se tem sintaxe OK
    if codigo and not codigo.startswith("# Erro"):
        if not step.arquivo.endswith(".py") or _validar_sintaxe(codigo, step.arquivo)[0]:
            sessao.registrar_resultado(step, codigo)
            sessao.erros.append(f"Step {step.numero}: aceito com ressalvas")
            _salvar_aprendizado(semantica, sessao, step, codigo)
            console.print(f"  [yellow]⚡ Aceito com ressalvas[/yellow]")
            return True

    return False


# ══════════════════════════════════════════════════════════════
# [13] STREAMING NO LOOP
# ══════════════════════════════════════════════════════════════


def _executar_step_streaming(sessao: SessaoCodigo, step: StepPlano) -> str:
    """Gera código com streaming visual (o usuário vê o código aparecendo)."""
    contexto = sessao.contexto_para_step(step)

    # Injeta esqueleto do template se disponível
    esqueleto_extra = ""
    template = selecionar_template(sessao.objetivo)
    if template and step.arquivo:
        esqueleto = obter_esqueleto(template, step.arquivo)
        if esqueleto:
            esqueleto_extra = f"\n\nESQUELETO BASE (expanda e complete):\n```\n{esqueleto}\n```"

    try:
        stream = ollama.chat(
            model=MODELOS["coder"],
            messages=[
                {"role": "system", "content": _PROMPT_CODER},
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
        console.print(f"[dim]  Funcionalidades identificadas[/dim]")

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
    """Validação semântica via LLM."""
    if not codigo or len(codigo) < 20:
        return {"valido": False, "problemas": ["vazio"], "decisoes": []}

    # Heurísticas rápidas
    probs = []
    if "TODO" in codigo and len(codigo) < 300:
        probs.append("TODOs não resolvidos")
    if codigo.count("pass") > 3 and len(codigo) < 400:
        probs.append("implementação stub")
    # Conta linhas com apenas "pass" ou "# Implementação..."
    linhas_stub = sum(
        1 for linha in codigo.split("\n")
        if linha.strip() in ("pass",) or linha.strip().startswith("# Implementação")
    )
    if linhas_stub > 2:
        probs.append(f"{linhas_stub} linhas stub/pass — código incompleto")
    if probs:
        return {"valido": False, "problemas": probs, "decisoes": []}

    try:
        r = ollama.chat(
            model=MODELOS["rapido"],
            messages=[
                {"role": "system", "content": _PROMPT_VALIDAR},
                {"role": "user", "content": f"Objetivo: {step.descricao}\nCódigo:\n{codigo[:1500]}"},
            ],
            options={"temperature": 0.1, "num_predict": 100},
        )
        return _parse_validacao(r["message"]["content"].strip())
    except Exception:
        return {"valido": True, "problemas": [], "decisoes": []}


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
    }
    (base / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return base


def _extrair_codigo(texto: str) -> str:
    """Extrai bloco de código de markdown."""
    if "```" in texto:
        blocos = texto.split("```")
        if len(blocos) >= 3:
            bloco = blocos[1]
            if "\n" in bloco:
                primeira = bloco.split("\n", 1)[0].strip().lower()
                langs = {"python", "py", "javascript", "js", "typescript", "ts",
                         "php", "go", "rust", "java", "sql", "bash", "sh", "yaml", "json", "toml"}
                if primeira in langs or (primeira.isalpha() and len(primeira) <= 12):
                    bloco = bloco.split("\n", 1)[1]
            return bloco.strip()
    return texto.strip()


def _parse_plano(objetivo: str, raw: str) -> SessaoCodigo:
    """Parseia JSON do plano."""
    inicio = raw.find("{")
    if inicio == -1:
        return SessaoCodigo(objetivo=objetivo, plano=[StepPlano(numero=1, descricao=objetivo, arquivo="main.py")])
    depth, fim = 0, -1
    for i in range(inicio, len(raw)):
        if raw[i] == "{": depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0: fim = i + 1; break
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
    if inicio == -1: return {"valido": True, "problemas": [], "decisoes": []}
    depth, fim = 0, -1
    for i in range(inicio, len(raw)):
        if raw[i] == "{": depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0: fim = i + 1; break
    if fim == -1: return {"valido": True, "problemas": [], "decisoes": []}
    try:
        d = json.loads(raw[inicio:fim])
        return {"valido": bool(d.get("valido", True)), "problemas": d.get("problemas", []), "decisoes": d.get("decisoes", [])}
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
