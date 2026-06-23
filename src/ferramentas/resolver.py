"""
Ferramentas que executam ANTES do LLM (Nível 1).
O LLM só interpreta o resultado — não faz o cálculo.
"""

import ast
import math
import operator
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from src.core.config import BASE_DIR

TIMEOUT_COMANDO_S = 20
MAX_SAIDA_CHARS = 4000

# ══════════════════════════════════════════════════════════════
# AVALIADOR MATEMÁTICO SEGURO (sem eval)
# ══════════════════════════════════════════════════════════════

_OPERADORES_BINARIOS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_OPERADORES_UNARIOS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCOES_SEGURAS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "pow": pow,
    "abs": abs,
    "round": round,
}

_CONSTANTES_SEGURAS = {
    "pi": math.pi,
    "e": math.e,
}


def _avaliar_ast(node: ast.AST) -> float | int:
    """Avalia recursivamente um nó AST com whitelist de operações."""
    if isinstance(node, ast.Expression):
        return _avaliar_ast(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Constante não numérica: {node.value}")

    if isinstance(node, ast.Name):
        nome = node.id.lower()
        if nome in _CONSTANTES_SEGURAS:
            return _CONSTANTES_SEGURAS[nome]
        raise ValueError(f"Variável não permitida: {node.id}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERADORES_BINARIOS:
            raise ValueError(f"Operador não permitido: {op_type.__name__}")
        esquerda = _avaliar_ast(node.left)
        direita = _avaliar_ast(node.right)
        # Proteção contra potências absurdas
        if op_type is ast.Pow and isinstance(direita, (int, float)) and direita > 1000:
            raise ValueError("Expoente muito grande")
        return _OPERADORES_BINARIOS[op_type](esquerda, direita)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPERADORES_UNARIOS:
            raise ValueError(f"Operador unário não permitido: {op_type.__name__}")
        return _OPERADORES_UNARIOS[op_type](_avaliar_ast(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Chamada de função inválida")
        nome = node.func.id.lower()
        if nome not in _FUNCOES_SEGURAS:
            raise ValueError(f"Função não permitida: {nome}")
        args = [_avaliar_ast(a) for a in node.args]
        return _FUNCOES_SEGURAS[nome](*args)

    raise ValueError(f"Nó AST não suportado: {type(node).__name__}")


def _eval_seguro(expressao: str) -> float | int:
    """Avalia expressão matemática usando AST — sem eval()."""
    tree = ast.parse(expressao, mode="eval")
    return _avaliar_ast(tree)


# ══════════════════════════════════════════════════════════════
# COMANDOS PERMITIDOS (allowlist)
# ══════════════════════════════════════════════════════════════

COMANDOS_PERMITIDOS = {
    "ls", "cat", "head", "tail", "wc", "echo", "pwd", "find", "grep",
    "tree", "file", "du", "df", "which", "whoami", "date", "uname",
    "pip", "pip3", "node", "npm", "git",
    "jq", "sed", "awk", "sort", "uniq", "diff",
}

# Subcomandos perigosos mesmo em binários permitidos
_SUBCMD_BLOQUEADOS = {
    "git": {"push", "remote", "config"},
    "pip": {"install", "uninstall"},
    "pip3": {"install", "uninstall"},
    "npm": {"run", "exec", "install", "uninstall", "publish"},
}

_PREFIXOS_CONFIRMACAO = ("confirmar:", "confirmo:", "execute:")


def remover_confirmacao(texto: str) -> tuple[str, bool]:
    """Remove prefixo de confirmação explícita e informa se ele estava presente."""
    texto_limpo = texto.strip()
    texto_lower = texto_limpo.lower()
    for prefixo in _PREFIXOS_CONFIRMACAO:
        if texto_lower.startswith(prefixo):
            return texto_limpo[len(prefixo):].lstrip(), True
    return texto, False


def descrever_acao_local_mutavel(texto: str) -> str | None:
    """Identifica ações locais com efeito colateral antes de executá-las."""
    texto_sem_confirmacao, _ = remover_confirmacao(texto)
    base = texto_sem_confirmacao.strip().lower()
    if base.startswith(("criar arquivo ", "crie arquivo ")):
        return "criação ou sobrescrita de arquivo"
    if re.match(r"^(?:executar|rode|rodar)\s+comando\s+.+$", base):
        return "execução de comando local"
    return None


def _resolver_caminho(caminho: str) -> Path:
    """Resolve caminho relativo ao projeto e impede acesso fora da raiz."""
    caminho = caminho.strip().strip('"').strip("'")
    alvo = (BASE_DIR / caminho).resolve() if not Path(caminho).is_absolute() else Path(caminho).resolve()
    base = BASE_DIR.resolve()
    if not alvo.is_relative_to(base):
        raise ValueError("Acesso negado: caminho fora da pasta do projeto.")
    return alvo


def listar_pasta(caminho: str) -> str:
    """Lista arquivos e pastas de um diretório."""
    try:
        alvo = _resolver_caminho(caminho or ".")
        if not alvo.exists():
            return f"Pasta não encontrada: {alvo}"
        if not alvo.is_dir():
            return f"Caminho não é pasta: {alvo}"

        itens = sorted(alvo.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if not itens:
            return f"Pasta vazia: {alvo.relative_to(BASE_DIR)}"

        linhas = [f"Conteúdo de {alvo.relative_to(BASE_DIR)}:"]
        for item in itens[:200]:
            sufixo = "/" if item.is_dir() else ""
            linhas.append(f"- {item.name}{sufixo}")
        if len(itens) > 200:
            linhas.append(f"... ({len(itens) - 200} itens omitidos)")
        return "\n".join(linhas)
    except Exception as e:
        return f"Erro ao listar pasta: {e}"


def ler_arquivo(caminho: str) -> str:
    """Lê conteúdo de um arquivo texto dentro do projeto."""
    try:
        alvo = _resolver_caminho(caminho)
        if not alvo.exists():
            return f"Arquivo não encontrado: {alvo}"
        if not alvo.is_file():
            return f"Caminho não é arquivo: {alvo}"

        conteudo = alvo.read_text(encoding="utf-8")
        if len(conteudo) > MAX_SAIDA_CHARS:
            conteudo = conteudo[:MAX_SAIDA_CHARS] + "\n... (truncado)"
        return f"Arquivo: {alvo.relative_to(BASE_DIR)}\n\n{conteudo}"
    except Exception as e:
        return f"Erro ao ler arquivo: {e}"


def criar_arquivo(caminho: str, conteudo: str) -> str:
    """Cria (ou sobrescreve) arquivo dentro da pasta do projeto."""
    try:
        alvo = _resolver_caminho(caminho)
        alvo.parent.mkdir(parents=True, exist_ok=True)
        alvo.write_text(conteudo, encoding="utf-8")
        tamanho = len(conteudo)
        return f"Arquivo salvo: {alvo.relative_to(BASE_DIR)} ({tamanho} chars)"
    except Exception as e:
        return f"Erro ao criar arquivo: {e}"


def executar_comando_local(comando: str) -> str:
    """Executa comando shell no diretório do projeto com allowlist."""
    comando_limpo = comando.strip()

    # Parse seguro para extrair o binário principal
    try:
        partes = shlex.split(comando_limpo)
    except ValueError:
        return "Erro: comando com sintaxe inválida (aspas não fechadas, etc)."

    if not partes:
        return "Comando vazio."

    binario = Path(partes[0]).name  # pega apenas o nome, sem path absoluto

    if binario not in COMANDOS_PERMITIDOS:
        return (
            f"Comando '{binario}' não está na lista de permitidos.\n"
            f"Permitidos: {', '.join(sorted(COMANDOS_PERMITIDOS))}"
        )

    # Bloqueio de subcomandos perigosos
    if binario in _SUBCMD_BLOQUEADOS and len(partes) > 1:
        subcmd = partes[1].lower()
        if subcmd in _SUBCMD_BLOQUEADOS[binario]:
            return f"Subcomando '{binario} {subcmd}' bloqueado por segurança."

    # Bloqueio de flags perigosas mesmo em comandos permitidos
    flags_perigosas = ["--force", "-rf", "--hard", "--no-preserve-root"]
    if any(f in partes for f in flags_perigosas):
        return "Flags destrutivas bloqueadas por segurança."

    try:
        proc = subprocess.run(
            partes,  # lista de args — sem shell=True
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_COMANDO_S,
        )
        saida = (proc.stdout or "").strip()
        erro = (proc.stderr or "").strip()
        resumo = [f"Comando: {comando_limpo}", f"Exit code: {proc.returncode}"]
        if saida:
            resumo.append("\nSTDOUT:\n" + saida[:MAX_SAIDA_CHARS])
        if erro:
            resumo.append("\nSTDERR:\n" + erro[:MAX_SAIDA_CHARS])
        return "\n".join(resumo)
    except subprocess.TimeoutExpired:
        return f"Comando excedeu timeout de {TIMEOUT_COMANDO_S}s."
    except FileNotFoundError:
        return f"Comando '{binario}' não encontrado no sistema."
    except Exception as e:
        return f"Erro ao executar comando: {e}"


def verificar_ferramenta_sistema(texto: str) -> str | None:
    """Interpreta comandos de sistema em linguagem natural.

    Formatos suportados:
    - listar pasta <caminho>
    - ler arquivo <caminho>
    - criar arquivo <caminho> ::: <conteudo>
    - executar comando <comando>
    """
    texto_limpo = texto.strip()
    texto_lower = texto_limpo.lower()

    m = re.match(r"^(?:listar|ler)\s+(?:pasta|diretorio|diretório)\s*(.*)$", texto_lower)
    if m:
        caminho = texto_limpo[m.start(1):].strip() or "."
        return listar_pasta(caminho)

    m = re.match(r"^ler\s+arquivo\s+(.+)$", texto_lower)
    if m:
        caminho = texto_limpo[m.start(1):].strip()
        return ler_arquivo(caminho)

    if texto_lower.startswith("criar arquivo ") or texto_lower.startswith("crie arquivo "):
        # Usa delimitador simples para permitir conteúdo multilinha.
        if ":::" not in texto_limpo:
            return (
                "Formato para criar arquivo: criar arquivo <caminho> ::: <conteudo>.\n"
                "Exemplo: criar arquivo notas/todo.txt ::: primeira linha"
            )
        prefixo, conteudo = texto_limpo.split(":::", 1)
        caminho = re.sub(r"^(criar|crie)\s+arquivo\s+", "", prefixo, flags=re.IGNORECASE).strip()
        if not caminho:
            return "Informe o caminho do arquivo."
        return criar_arquivo(caminho, conteudo.lstrip("\n"))

    m = re.match(r"^(?:executar|rode|rodar)\s+comando\s+(.+)$", texto_lower)
    if m:
        comando = texto_limpo[m.start(1):].strip()
        return executar_comando_local(comando)

    return None


def calcular(expressao: str) -> str | None:
    """Avalia expressões matemáticas de forma segura (sem eval)."""
    expressao_limpa = expressao.strip()

    padrao_math = re.compile(
        r'^[\d\s\+\-\*\/\.\(\)\%\^]+$|'
        r'\b(sqrt|sin|cos|tan|log|pow|abs|round|pi|e)\b'
    )

    if not padrao_math.search(expressao_limpa):
        return None

    expressao_limpa = expressao_limpa.replace("^", "**")

    try:
        resultado = _eval_seguro(expressao_limpa)
        return f"Resultado: {resultado}"
    except Exception:
        return None


def obter_data_hora(timezone: str | None = None) -> str:
    """
    Retorna data e hora formatada.
    Se timezone for fornecido, calcula para aquele fuso.
    Usa apenas stdlib (zoneinfo, disponível desde Python 3.9).
    """
    try:
        if timezone:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(timezone)
            agora = datetime.now(tz)
            return (
                f"Data: {agora.strftime('%d/%m/%Y')} "
                f"({agora.strftime('%A')})\n"
                f"Hora: {agora.strftime('%H:%M:%S')}\n"
                f"Fuso: {timezone} (UTC{agora.strftime('%z')})"
            )
    except (KeyError, ImportError):
        pass

    agora = datetime.now()
    return (
        f"Data: {agora.strftime('%d/%m/%Y')} "
        f"({agora.strftime('%A')})\n"
        f"Hora: {agora.strftime('%H:%M:%S')}"
    )


# Mapa de locais comuns → timezone IANA (sem deps externas)
_LOCAIS_TIMEZONE: dict[str, str] = {
    # África
    "congo": "Africa/Kinshasa",
    "kinshasa": "Africa/Kinshasa",
    "brazzaville": "Africa/Brazzaville",
    "lagos": "Africa/Lagos",
    "nigeria": "Africa/Lagos",
    "cairo": "Africa/Cairo",
    "egito": "Africa/Cairo",
    "johannesburg": "Africa/Johannesburg",
    "africa do sul": "Africa/Johannesburg",
    "nairobi": "Africa/Nairobi",
    "quenia": "Africa/Nairobi",
    # Américas
    "nova york": "America/New_York",
    "new york": "America/New_York",
    "los angeles": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "sao paulo": "America/Sao_Paulo",
    "são paulo": "America/Sao_Paulo",
    "brasil": "America/Sao_Paulo",
    "brasilia": "America/Sao_Paulo",
    "rio de janeiro": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires",
    "argentina": "America/Argentina/Buenos_Aires",
    "mexico": "America/Mexico_City",
    "bogota": "America/Bogota",
    "colombia": "America/Bogota",
    "lima": "America/Lima",
    "peru": "America/Lima",
    "santiago": "America/Santiago",
    "chile": "America/Santiago",
    "toronto": "America/Toronto",
    "canada": "America/Toronto",
    "vancouver": "America/Vancouver",
    # Europa
    "londres": "Europe/London",
    "london": "Europe/London",
    "inglaterra": "Europe/London",
    "paris": "Europe/Paris",
    "franca": "Europe/Paris",
    "berlim": "Europe/Berlin",
    "alemanha": "Europe/Berlin",
    "madrid": "Europe/Madrid",
    "espanha": "Europe/Madrid",
    "roma": "Europe/Rome",
    "italia": "Europe/Rome",
    "lisboa": "Europe/Lisbon",
    "portugal": "Europe/Lisbon",
    "moscou": "Europe/Moscow",
    "russia": "Europe/Moscow",
    "amsterdam": "Europe/Amsterdam",
    "holanda": "Europe/Amsterdam",
    # Ásia
    "tokyo": "Asia/Tokyo",
    "toquio": "Asia/Tokyo",
    "japao": "Asia/Tokyo",
    "pequim": "Asia/Shanghai",
    "beijing": "Asia/Shanghai",
    "china": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "dubai": "Asia/Dubai",
    "emirados": "Asia/Dubai",
    "india": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata",
    "nova delhi": "Asia/Kolkata",
    "seul": "Asia/Seoul",
    "coreia": "Asia/Seoul",
    "singapura": "Asia/Singapore",
    "bangkok": "Asia/Bangkok",
    "tailandia": "Asia/Bangkok",
    # Oceania
    "sydney": "Australia/Sydney",
    "australia": "Australia/Sydney",
    "auckland": "Pacific/Auckland",
    "nova zelandia": "Pacific/Auckland",
}


def _extrair_local_hora(texto: str) -> str | None:
    """
    Detecta se a pergunta pede hora de um local específico.
    Retorna o timezone IANA ou None se for hora local.
    """
    texto_lower = texto.lower()

    # Padrões: "hora no Congo", "horas em Nova York", "horário de Tokyo"
    padrao = re.compile(
        r'(?:hora|horas|horário|horario|que horas)'
        r'.*?(?:em|no|na|nos|nas|do|da|de)\s+'
        r'([A-Za-zÀ-ú][A-Za-zÀ-ú\s]{1,30}?)(?:\?|$|,|\s*$)',
        re.IGNORECASE,
    )
    m = padrao.search(texto_lower)
    if m:
        local = m.group(1).strip().rstrip("?., ")
        # Busca no mapa
        if local in _LOCAIS_TIMEZONE:
            return _LOCAIS_TIMEZONE[local]
        # Tenta match parcial (ex: "estados unidos" → procura "nova york")
        for chave, tz in _LOCAIS_TIMEZONE.items():
            if chave in local or local in chave:
                return tz

    return None


def verificar_ferramenta_data(texto: str) -> str | None:
    """Verifica se a pergunta é sobre data/hora, incluindo fusos horários."""
    palavras_data = [
        "que horas", "hora atual", "que dia", "data de hoje",
        "dia hoje", "data atual", "horário", "que data",
        "horas em", "horas no", "horas na", "hora em", "hora no", "hora na",
        "horário em", "horário no", "horário na", "horário de", "horário do",
    ]
    texto_lower = texto.lower()

    detectou = False
    for p in palavras_data:
        if p in texto_lower:
            detectou = True
            break

    if not detectou:
        return None

    # Tenta detectar local específico
    timezone = _extrair_local_hora(texto)
    return obter_data_hora(timezone)


def verificar_ferramenta_calculo(texto: str) -> str | None:
    """Verifica se a pergunta é um cálculo simples."""
    padrao = re.compile(
        r'(?:quanto é|calcule?|resultado de|compute)\s*(.+)',
        re.IGNORECASE
    )
    match = padrao.search(texto)
    if match:
        return calcular(match.group(1))

    if re.match(r'^[\d\s\+\-\*\/\.\(\)\%\^]+$', texto.strip()):
        return calcular(texto)

    return None


def verificar_ferramenta_saudacao(texto: str) -> str | None:
    """Responde saudações curtas de forma determinística."""
    base = texto.strip().lower()
    saudacoes = {
        "oi", "olá", "ola", "e ai", "e aí", "bom dia", "boa tarde", "boa noite", "hello", "hey"
    }
    if base in saudacoes:
        return (
            "Olá! Estou pronto para ajudar. "
            "Você pode pedir código, análise, leitura/criação de arquivos e execução de comandos locais."
        )
    return None


def executar_ferramentas(texto: str) -> str | None:
    """
    Tenta resolver com ferramentas ANTES de enviar ao LLM.
    Retorna None se nenhuma ferramenta se aplica.
    """
    texto, _ = remover_confirmacao(texto)

    resultado = verificar_ferramenta_data(texto)
    if resultado:
        return resultado

    resultado = verificar_ferramenta_saudacao(texto)
    if resultado:
        return resultado

    resultado = verificar_ferramenta_calculo(texto)
    if resultado:
        return resultado

    resultado = verificar_ferramenta_sistema(texto)
    if resultado:
        return resultado

    return None
