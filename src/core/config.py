"""
Configuração central do sistema multiagente.
Otimizado para Mac M1 com 8GB RAM.

3 Níveis de Performance:
  Nível 1 (Turbo)   → Ferramentas + Cache + ChromaDB = SEM LLM
  Nível 2 (Rápido)  → Modelo pequeno (1.7B) com contexto mínimo
  Nível 3 (Profundo) → Modelo maior (4B) com RAG completo
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CAMINHOS
# ══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════
# PERFIS DE MODELOS
# ══════════════════════════════════════════════════════════════
# Escolha um perfil: "ultra_leve", "equilibrado" ou "maximo"
# Ultra-leve usa LFM2.5 → consome METADE da RAM dos Qwen

PERFIL_ATIVO = "ultra_leve"  # ← MUDE AQUI: "ultra_leve" | "equilibrado" | "maximo"

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
# AGENTES
# ══════════════════════════════════════════════════════════════

AGENTES = {
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
            "Analise os resultados de busca e apresente um resumo claro e objetivo. "
            "Cite fontes quando relevante."
        ),
        "palavras_chave": [
            "pesquisar", "buscar", "procurar", "encontrar", "notícias", "atualização",
            "novidade", "comparar", "alternativas", "melhor", "ranking", "tendência",
            "mercado", "preço", "custo", "ferramenta", "plataforma", "search",
        ],
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
# MEMÓRIA E CONTEXTO
# ══════════════════════════════════════════════════════════════

CONTEXTO_MAX_MENSAGENS = 3
CACHE_HABILITADO = True
CACHE_ARQUIVO = str(DATA_DIR / "cache.json")
MEMORIA_ARQUIVO = str(DATA_DIR / "memoria.db")

# ChromaDB - Memória Semântica
CHROMADB_DIR = str(DATA_DIR / "chromadb")
CHROMADB_COLLECTION = "conversas"
CHROMADB_TOP_K = 6
CHROMADB_THRESHOLD = 0.62
CHROMADB_NIVEL1_THRESHOLD = 0.86
RAG_MAX_DOCS = 4
RAG_MAX_CHARS = 2200
EMBEDDING_MODEL = MODELOS["embedding"]

# ══════════════════════════════════════════════════════════════
# COORDENADOR
# ══════════════════════════════════════════════════════════════

COORDENADOR_MODELO = MODELOS["coordenador"]
COORDENADOR_SYSTEM = (
    "Você é um coordenador. Sua ÚNICA tarefa é classificar a pergunta do usuário. "
    "Responda APENAS com o nome do agente mais adequado: programador, pesquisador ou analista. "
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
