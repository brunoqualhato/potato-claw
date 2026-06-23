"""
Configuração central do sistema multiagente.
Otimizado para Mac M1 com 8GB RAM.

3 Níveis de Performance:
  Nível 1 (Turbo)   → Ferramentas + Cache + ChromaDB = SEM LLM
  Nível 2 (Rápido)  → Modelo pequeno (1.7B) com contexto mínimo
  Nível 3 (Profundo) → Modelo maior (4B) com RAG completo
"""

import os
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CARREGA .env (sem dependência externa — leve para 8GB)
# ══════════════════════════════════════════════════════════════

def _carregar_env(caminho: Path):
    """Carrega .env manualmente — zero dependências extras."""
    if not caminho.exists():
        return
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        if "=" not in linha:
            continue
        chave, _, valor = linha.partition("=")
        chave = chave.strip()
        valor = valor.strip().strip('"').strip("'")
        os.environ.setdefault(chave, valor)


# ══════════════════════════════════════════════════════════════
# CAMINHOS
# ══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent.parent
_carregar_env(BASE_DIR / ".env")

DATA_DIR = Path(os.environ.get("NEURON_DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════
# PERFIS DE MODELOS
# ══════════════════════════════════════════════════════════════
# Escolha um perfil: "ultra_leve", "equilibrado" ou "maximo"
# Ultra-leve usa LFM2.5 → consome METADE da RAM dos Qwen

PERFIL_ATIVO = os.environ.get("NEURON_PERFIL", "ultra_leve")

PERFIS = {
    # ─── ULTRA-LEVE: LFM2.5 (melhor para 8GB) ───
    # Pico máximo: ~1.5 GB. Sobra RAM pra tudo.
    "ultra_leve": {
        "coordenador": "LiquidAI/lfm2.5-1.2b-instruct",  # 731 MB - roteamento
        "rapido": "LiquidAI/lfm2.5-1.2b-instruct",       # 731 MB - nível 2
        "coder": "qwen2.5-coder:3b",                      # 2.5 GB - código (nível 3)
        "completo": "maternion/lfm2.5",                    # ~1.5 GB MoE 8B-A1B - nível 3 geral
        "embedding": "nomic-embed-text",                   # 270 MB
    },

    # ─── EQUILIBRADO: Mix LFM + Qwen ───
    # Pico máximo: ~3.2 GB.
    "equilibrado": {
        "coordenador": "LiquidAI/lfm2.5-1.2b-instruct",  # 731 MB
        "rapido": "qwen3:1.7b",                           # 1.5 GB
        "coder": "qwen2.5-coder:3b",                      # 2.5 GB
        "completo": "qwen3:4b",                           # 3.2 GB
        "embedding": "nomic-embed-text",                   # 270 MB
    },

    # ─── MÁXIMO: Qwen puro (mais qualidade, mais RAM) ───
    # Pico máximo: ~3.5 GB.
    "maximo": {
        "coordenador": "qwen3:1.7b",                      # 1.5 GB
        "rapido": "qwen3:1.7b",                           # 1.5 GB
        "coder": "qwen2.5-coder:3b",                      # 2.5 GB
        "completo": "qwen3:4b",                           # 3.2 GB
        "embedding": "nomic-embed-text",                   # 270 MB
    },
}

# Modelos ativos (baseado no perfil selecionado)
MODELOS = PERFIS[PERFIL_ATIVO]

# Override opcional: forca um unico modelo em todos os papeis de chat (mantem o
# embedding). Util para hardware com um modelo so ou para testes rapidos.
# Ex.: NEURON_MODELO_UNICO=qwen2.5:0.5b
_MODELO_UNICO = os.environ.get("NEURON_MODELO_UNICO")
if _MODELO_UNICO:
    MODELOS = {
        papel: (modelo if papel == "embedding" else _MODELO_UNICO)
        for papel, modelo in MODELOS.items()
    }

# ══════════════════════════════════════════════════════════════
# NÍVEIS DE PERFORMANCE
# ══════════════════════════════════════════════════════════════

NIVEIS = {
    1: {
        "nome": "Turbo",
        "descricao": "Ferramentas + Cache + ChromaDB (sem LLM)",
        "modelo": None,
        "max_tokens": 0,
    },
    2: {
        "nome": "Rápido",
        "descricao": f"Modelo leve ({MODELOS['rapido']}) + contexto curto",
        "modelo": MODELOS["rapido"],
        "max_tokens": 512,
        "temperatura": 0.4,
        "contexto_msgs": 2,
    },
    3: {
        "nome": "Profundo",
        "descricao": f"Modelo completo ({MODELOS['completo']}) + RAG ChromaDB",
        "modelo": MODELOS["completo"],
        "max_tokens": 2048,
        "temperatura": 0.7,
        "contexto_msgs": 5,
    },
}

# ══════════════════════════════════════════════════════════════
# INFERÊNCIA OLLAMA - tuning para hardware fraco ("PC de batata")
# ══════════════════════════════════════════════════════════════
# Em máquinas modestas (ex.: M1 8GB) o que mais trava é:
#   1. vários modelos residentes ao mesmo tempo (coordenador + embedding +
#      rápido/profundo) estourando a RAM e indo para swap;
#   2. KV cache sem teto (num_ctx grande) consumindo RAM por requisição.
# Os parâmetros abaixo atacam os dois pontos. Tudo é sobreescrevível por env.

def _detectar_ram_gb() -> float:
    """RAM total em GB, sem depender de psutil (mantém o projeto leve)."""
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for linha in meminfo.read_text().splitlines():
                if linha.startswith("MemTotal:"):
                    return int(linha.split()[1]) / (1024 * 1024)
        import subprocess
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


RAM_GB = round(_detectar_ram_gb(), 1)

# Threads: deixa 1 núcleo livre para o SO não engasgar em máquina fraca.
NUM_THREAD = int(os.environ.get("NEURON_NUM_THREAD", str(max(2, (os.cpu_count() or 4) - 1))))

# keep_alive: "efêmero" solta da RAM os modelos auxiliares logo após o uso
# (coordenador, analisador, resumo); "principal" segura o modelo de resposta
# por alguns minutos para não recarregar a cada turno.
KEEP_ALIVE_EFEMERO = os.environ.get("NEURON_KEEP_ALIVE_EFEMERO", "0s")
KEEP_ALIVE_PRINCIPAL = os.environ.get("NEURON_KEEP_ALIVE_PRINCIPAL", "5m")
WARMUP_HABILITADO = os.environ.get("NEURON_WARMUP_HABILITADO", "false").lower() in ("true", "1", "yes")
WARMUP_MODELOS = os.environ.get("NEURON_WARMUP_MODELOS", "rapido")

# Timeout duro para não pendurar o processo quando o modelo trava sob pressão de RAM.
OLLAMA_TIMEOUT = int(os.environ.get("NEURON_OLLAMA_TIMEOUT", "120"))

# num_ctx por nível (janela de contexto = KV cache = RAM). Menor no nível rápido.
NUM_CTX_NIVEL = {
    2: int(os.environ.get("NEURON_NUM_CTX_RAPIDO", "2048")),
    3: int(os.environ.get("NEURON_NUM_CTX_PROFUNDO", "4096")),
}
# Janela mínima para chamadas auxiliares (classificação de intenção, resumo).
NUM_CTX_AUXILIAR = int(os.environ.get("NEURON_NUM_CTX_AUXILIAR", "1024"))
CONTEXTO_MAX_MSGS = int(os.environ.get("NEURON_CONTEXTO_MAX_MSGS", "5"))

# Ações que alteram arquivos ou executam comandos exigem confirmação explícita.
# Use apenas em ambientes controlados/automatizados.
AUTO_APROVAR_ACOES_LOCAIS = os.environ.get(
    "NEURON_AUTO_APROVAR_ACOES_LOCAIS", "false"
).lower() in ("true", "1", "yes")


def perfil_sugerido_por_ram() -> str | None:
    """Sugere um perfil pelo total de RAM. Não força nada (respeita NEURON_PERFIL)."""
    if not RAM_GB:
        return None
    if RAM_GB < 9:
        return "ultra_leve"
    if RAM_GB < 17:
        return "equilibrado"
    return "maximo"


# ══════════════════════════════════════════════════════════════
# AGENTES
# ══════════════════════════════════════════════════════════════

AGENTES = {
    "generalista": {
        "modelo_rapido": MODELOS["rapido"],
        "modelo_profundo": MODELOS["rapido"],
        "system_prompt": (
            "Você é um assistente generalista eficiente para dúvidas abertas e mensagens informais. "
            "Responda curto, claro e objetivo. "
            "Se a pergunta exigir especialização técnica profunda, diga isso e peça contexto adicional."
        ),
        "palavras_chave": [],
        "nivel_preferido": 2,
    },
    "programador": {
        "modelo_rapido": MODELOS["rapido"],
        "modelo_profundo": MODELOS["coder"],
        "system_prompt": (
            "Você é um programador especialista em Python, AWS, Laravel, Angular e IA. "
            "Responda de forma direta e prática com código funcional. "
            "Use boas práticas e explique brevemente o que faz."
        ),
        "palavras_chave": [
            "python", "código", "programar", "função", "classe", "bug", "erro",
            "aws", "lambda", "s3", "ec2", "laravel", "php", "angular", "typescript",
            "docker", "api", "banco", "sql", "git", "deploy", "implementar",
            "script", "automação", "refatorar", "teste", "debug", "coder",
        ],
        "nivel_preferido": 3,
    },
    "pesquisador": {
        "modelo_rapido": MODELOS["rapido"],
        "modelo_profundo": MODELOS["completo"],
        "system_prompt": (
            "Você é um pesquisador especialista em buscar informações atualizadas na web. "
            "Quando resultados de pesquisa forem fornecidos no contexto, USE-OS como fonte primária. "
            "NUNCA diga que não tem acesso a dados em tempo real — os dados já estão no contexto. "
            "Apresente um resumo claro e objetivo, citando fontes e incluindo links úteis."
        ),
        "palavras_chave": [
            # Busca genérica
            "pesquisar", "buscar", "procurar", "encontrar", "notícias", "atualização",
            "novidade", "comparar", "alternativas", "melhor", "ranking", "tendência",
            "mercado", "preço", "custo", "ferramenta", "plataforma", "search",
            # Clima e tempo real
            "temperatura", "clima", "tempo em", "previsão", "chuva", "sol", "vento",
            "calor", "frio", "umidade", "sensação térmica",
            # Cotações e finanças
            "cotação", "dólar", "euro", "bitcoin", "cripto", "câmbio", "bolsa",
            "ações", "ibovespa", "nasdaq", "inflação", "selic",
            # Documentação e referências
            "documentação", "docs", "doc", "manual", "referência", "how to",
            "tutorial", "guia", "sintaxe", "como usar", "como instalar", "como configurar",
            # Notícias e eventos atuais
            "hoje", "agora", "atual", "recente", "último", "nova versão", "release",
            "resultado", "placar", "jogo", "campeonato", "eleição",
            # Versão e lançamentos
            "última versão", "versão atual", "versão do", "versão de", "qual versão",
            "lançamento", "atualização do", "atualização de",
        ],
        "usa_web": True,
        "nivel_preferido": 2,
    },
    "analista": {
        "modelo_rapido": MODELOS["rapido"],
        "modelo_profundo": MODELOS["completo"],
        "system_prompt": (
            "Você é um analista de dados e negócios. "
            "Analise informações, identifique padrões e forneça insights acionáveis. "
            "Use raciocínio estruturado e apresente conclusões claras."
        ),
        "palavras_chave": [
            "analisar", "análise", "dados", "métrica", "relatório", "dashboard",
            "estratégia", "plano", "negócio", "saas", "voip", "viabilidade",
            "projeção", "estimativa", "risco", "oportunidade", "decisão",
        ],
        "nivel_preferido": 3,
    },
}

# ══════════════════════════════════════════════════════════════
# CANAIS (hub de conexoes) - opt-in via env, offline-first
# ══════════════════════════════════════════════════════════════

def canais_configurados() -> list[dict]:
    """Canais habilitados via variaveis de ambiente.

    Sem nenhuma var definida, retorna [] (modo offline puro: so CLI).
    Exemplo: TELEGRAM_BOT_TOKEN=123:abc TELEGRAM_ALLOW_LIST=111,222
    """
    canais: list[dict] = []
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if tg_token:
        allow = os.environ.get("TELEGRAM_ALLOW_LIST", "")
        canais.append({
            "tipo": "telegram",
            "token": tg_token,
            "allow_list": [x.strip() for x in allow.split(",") if x.strip()],
        })
    return canais


# ══════════════════════════════════════════════════════════════
# MEMÓRIA E CONTEXTO
# ══════════════════════════════════════════════════════════════

CACHE_HABILITADO = os.environ.get("NEURON_CACHE_HABILITADO", "true").lower() in ("true", "1", "yes")
CACHE_ARQUIVO = str(DATA_DIR / "cache.json")
MEMORIA_ARQUIVO = str(DATA_DIR / "memoria.db")

# ChromaDB - Memória Semântica
CHROMADB_DIR = str(DATA_DIR / "chromadb")
CHROMADB_COLLECTION = "conversas"
CHROMADB_TOP_K = int(os.environ.get("NEURON_CHROMADB_TOP_K", "6"))
CHROMADB_THRESHOLD = float(os.environ.get("NEURON_CHROMADB_THRESHOLD", "0.62"))
CHROMADB_NIVEL1_THRESHOLD = float(os.environ.get("NEURON_CHROMADB_NIVEL1_THRESHOLD", "0.86"))
RAG_MAX_DOCS = int(os.environ.get("NEURON_RAG_MAX_DOCS", "4"))
RAG_MAX_CHARS = int(os.environ.get("NEURON_RAG_MAX_CHARS", "2200"))
EMBEDDING_MODEL = MODELOS["embedding"]

# Web RAG — Pipeline de busca profunda
WEB_RAG_MAX_PAGINAS = int(os.environ.get("NEURON_WEB_RAG_MAX_PAGINAS", "3"))
WEB_RAG_FETCH_TIMEOUT = int(os.environ.get("NEURON_WEB_RAG_FETCH_TIMEOUT", "8"))
WEB_RAG_MAX_MD_CHARS = int(os.environ.get("NEURON_WEB_RAG_MAX_MD_CHARS", "6000"))
WEB_RAG_CACHE_TTL = int(os.environ.get("NEURON_WEB_RAG_CACHE_TTL", "3600"))

# ══════════════════════════════════════════════════════════════
# COORDENADOR
# ══════════════════════════════════════════════════════════════

COORDENADOR_MODELO = MODELOS["coordenador"]
COORDENADOR_SYSTEM = (
    "Você é um coordenador. Sua ÚNICA tarefa é classificar a pergunta do usuário. "
    "Responda APENAS com o nome do agente mais adequado: generalista, programador, pesquisador ou analista. "
    "Nada mais. Apenas o nome."
)

# ══════════════════════════════════════════════════════════════
# CLASSIFICADOR DE COMPLEXIDADE
# ══════════════════════════════════════════════════════════════

INDICADORES_SIMPLES = [
    "o que é", "defina", "explique brevemente", "resuma",
    "sim ou não", "qual a diferença", "liste", "enumere",
]

INDICADORES_COMPLEXOS = [
    "implemente", "crie", "desenvolva", "arquitetura", "projete",
    "analise detalhadamente", "compare", "avalie", "otimize",
    "refatore", "debug", "por que", "como funciona internamente",
    "passo a passo", "com exemplo", "código completo",
]
