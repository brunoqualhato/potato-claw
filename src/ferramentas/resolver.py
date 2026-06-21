"""
Ferramentas que executam ANTES do LLM (Nível 1).
O LLM só interpreta o resultado — não faz o cálculo.
"""

import re
import math
import subprocess
from pathlib import Path
from datetime import datetime

from src.core.config import BASE_DIR


TIMEOUT_COMANDO_S = 20
MAX_SAIDA_CHARS = 4000


def _resolver_caminho(caminho: str) -> Path:
    """Resolve caminho relativo ao projeto e impede acesso fora da raiz."""
    caminho = caminho.strip().strip('"').strip("'")
    alvo = (BASE_DIR / caminho).resolve() if not Path(caminho).is_absolute() else Path(caminho).resolve()
    base = BASE_DIR.resolve()
    if not str(alvo).startswith(str(base)):
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
    """Executa comando shell no diretório do projeto com proteções básicas."""
    comando_limpo = comando.strip()
    comando_lower = comando_limpo.lower()

    bloqueados = [
        "rm -rf /",
        "shutdown",
        "reboot",
        "mkfs",
        "diskutil erase",
        "dd if=",
        ":(){",
    ]
    if any(token in comando_lower for token in bloqueados):
        return "Comando bloqueado por segurança."

    try:
        proc = subprocess.run(
            comando_limpo,
            shell=True,
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
    """Avalia expressões matemáticas simples."""
    expressao_limpa = expressao.strip()

    padrao_math = re.compile(
        r'^[\d\s\+\-\*\/\.\(\)\%\^]+$|'
        r'\b(sqrt|sin|cos|tan|log|pow|abs|round|pi|e)\b'
    )

    if not padrao_math.search(expressao_limpa):
        return None

    expressao_limpa = expressao_limpa.replace("^", "**")

    namespace = {
        "__builtins__": {},
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "pow": pow,
        "abs": abs, "round": round, "pi": math.pi, "e": math.e,
    }

    try:
        resultado = eval(expressao_limpa, namespace)
        return f"Resultado: {resultado}"
    except Exception:
        return None


def obter_data_hora() -> str:
    """Retorna data e hora atual formatada."""
    agora = datetime.now()
    return (
        f"Data: {agora.strftime('%d/%m/%Y')} "
        f"({agora.strftime('%A')})\n"
        f"Hora: {agora.strftime('%H:%M:%S')}"
    )


def verificar_ferramenta_data(texto: str) -> str | None:
    """Verifica se a pergunta é sobre data/hora."""
    palavras_data = [
        "que horas", "hora atual", "que dia", "data de hoje",
        "dia hoje", "data atual", "horário", "que data"
    ]
    texto_lower = texto.lower()
    for p in palavras_data:
        if p in texto_lower:
            return obter_data_hora()
    return None


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


def executar_ferramentas(texto: str) -> str | None:
    """
    Tenta resolver com ferramentas ANTES de enviar ao LLM.
    Retorna None se nenhuma ferramenta se aplica.
    """
    resultado = verificar_ferramenta_data(texto)
    if resultado:
        return resultado

    resultado = verificar_ferramenta_calculo(texto)
    if resultado:
        return resultado

    resultado = verificar_ferramenta_sistema(texto)
    if resultado:
        return resultado

    return None
