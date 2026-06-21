# 🤖 Sistema Multiagente Local — 3 Níveis de Performance

Sistema multiagente otimizado para **Mac M1 com 8GB RAM**.  
Python puro + ChromaDB para memória semântica.  
Suporte a **3 perfis de modelos**: ultra-leve (LFM2.5), equilibrado e máximo.

## Estrutura do Projeto

```
multi-agent/
├── main.py                    # Ponto de entrada (CLI)
├── requirements.txt           # Dependências
├── README.md
│
├── src/                       # Código fonte
│   ├── __init__.py
│   │
│   ├── core/                  # Núcleo do sistema
│   │   ├── __init__.py
│   │   ├── config.py          # Configuração central (perfis, modelos, níveis, agentes)
│   │   ├── llm.py             # Interface com Ollama
│   │   └── classificador.py   # Decide nível de performance (1/2/3)
│   │
│   ├── agentes/               # Lógica dos agentes
│   │   ├── __init__.py
│   │   ├── coordenador.py     # Roteia pergunta → agente correto
│   │   └── executor.py        # Motor de execução (pipeline 3 níveis)
│   │
│   ├── memoria/               # 3 camadas de memória
│   │   ├── __init__.py
│   │   ├── cache.py           # Camada 1: Cache exato (JSON)
│   │   ├── semantica.py       # Camada 2: ChromaDB (similaridade vetorial)
│   │   └── sqlite.py          # Camada 3: SQLite (histórico, resumos, métricas)
│   │
│   └── ferramentas/           # Ferramentas que rodam antes do LLM
│       ├── __init__.py
│       ├── resolver.py        # Cálculos, data/hora
│       └── web.py             # Pesquisa DuckDuckGo
│
└── data/                      # Dados persistentes (auto-criado)
    ├── cache.json             # Cache de respostas
    ├── memoria.db             # SQLite
    └── chromadb/              # Base vetorial
```

## Perfis de Modelos

O sistema oferece 3 perfis pré-configurados. Mude em `src/core/config.py`:

```python
PERFIL_ATIVO = "ultra_leve"  # ← "ultra_leve" | "equilibrado" | "maximo"
```

### ⚡ Ultra-leve (padrão) — LFM2.5 da Liquid AI

Ideal para **Mac M1 8GB**. Pico máximo de RAM: **~1.5 GB**.

| Função | Modelo | RAM | Nota |
|--------|--------|-----|------|
| Coordenador | `LiquidAI/lfm2.5-1.2b-instruct` | 731 MB | Roteamento ultra-rápido |
| Nível 2 (rápido) | `LiquidAI/lfm2.5-1.2b-instruct` | 731 MB | Respostas diretas |
| Nível 3 (profundo) | `maternion/lfm2.5` (8B-A1B MoE) | ~1.5 GB | Qualidade de 3-4B denso |
| Nível 3 (código) | `qwen2.5-coder:3b` | 2.5 GB | Só para código complexo |
| Embeddings | `nomic-embed-text` | 270 MB | ChromaDB vetorial |

**Por que LFM2.5?**
- Modelos da [Liquid AI](https://liquid.ai) otimizados para on-device
- **LFM2.5-1.2B**: cabe em 731 MB, context window de 125K tokens
- **LFM2.5-8B-A1B**: Mixture of Experts — 8B total mas **apenas 1B ativo por token**
  - Qualidade comparável a modelos densos de 3-4B
  - Velocidade superior ao Qwen 1.7B
  - Consome apenas ~1.5 GB

```bash
# Instalar modelos para perfil ultra-leve
ollama pull LiquidAI/lfm2.5-1.2b-instruct   # 731 MB
ollama pull maternion/lfm2.5                  # ~1.5 GB (MoE 8B-A1B)
ollama pull qwen2.5-coder:3b                  # 2.5 GB
ollama pull nomic-embed-text                  # 270 MB
```

### 🔄 Equilibrado — Mix LFM + Qwen

Pico máximo: **~3.2 GB**. Coordenador leve + modelos Qwen para respostas.

| Função | Modelo | RAM |
|--------|--------|-----|
| Coordenador | `LiquidAI/lfm2.5-1.2b-instruct` | 731 MB |
| Nível 2 | `qwen3:1.7b` | 1.5 GB |
| Nível 3 | `qwen3:4b` | 3.2 GB |
| Código | `qwen2.5-coder:3b` | 2.5 GB |

```bash
# Adicionar para perfil equilibrado
ollama pull qwen3:1.7b
ollama pull qwen3:4b
```

### 🧠 Máximo — Qwen puro

Pico máximo: **~3.5 GB**. Maior qualidade, mais RAM.

| Função | Modelo | RAM |
|--------|--------|-----|
| Coordenador | `qwen3:1.7b` | 1.5 GB |
| Nível 2 | `qwen3:1.7b` | 1.5 GB |
| Nível 3 | `qwen3:4b` | 3.2 GB |
| Código | `qwen2.5-coder:3b` | 2.5 GB |

```bash
# Adicionar para perfil máximo
ollama pull qwen3:1.7b
ollama pull qwen3:4b
```

## Arquitetura de 3 Níveis

```
                    Pergunta do Usuário
                           │
                    ┌──────▼──────┐
                    │ Classificar │
                    │Complexidade │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
   │  NÍVEL 1    │ │  NÍVEL 2    │ │  NÍVEL 3    │
   │   ⚡ Turbo   │ │  🚀 Rápido  │ │  🧠 Profundo │
   │             │ │             │ │             │
   │ • Cálculos  │ │ • LFM 1.2B  │ │ • LFM MoE   │
   │ • Data/hora │ │   ou Qwen   │ │   ou Qwen   │
   │ • Cache hit │ │ • 2 msgs    │ │ • ChromaDB  │
   │ • ChromaDB  │ │ • 512 tok   │ │   RAG       │
   │   (≥85%)    │ │ • Temp 0.4  │ │ • 5 msgs    │
   │             │ │             │ │ • 2048 tok  │
   │    0 MB     │ │  731MB-1.5GB│ │ 1.5GB-3.2GB │
   │    ~5ms     │ │   ~200ms    │ │   ~3-8s     │
   └─────────────┘ └─────────────┘ └─────────────┘
```

## Instalação Rápida

```bash
# 1. Ollama
brew install ollama

# 2. Modelos (perfil ultra-leve — recomendado para 8GB)
ollama pull LiquidAI/lfm2.5-1.2b-instruct
ollama pull maternion/lfm2.5
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text

# 3. Dependências Python
pip install -r requirements.txt

# 4. Executar
python main.py
```

## Comandos

| Comando | Descrição |
|---------|-----------|
| `/nivel <1\|2\|3>` | Forçar nível de performance |
| `/agente <nome>` | Forçar agente (programador, pesquisador, analista) |
| `/stats` | Métricas de performance por nível |
| `/knowledge <texto>` | Adicionar conhecimento ao ChromaDB |
| `/ingest <arquivo>` | Ingerir arquivo inteiro no ChromaDB |
| `/contexto` | Ver histórico recente |
| `/resumo` | Ver último resumo |
| `/limpar` | Limpar histórico |
| `/modelos` | Ver perfil ativo e status dos modelos |
| `/ajuda` | Lista de comandos |
| `/sair` | Encerrar |

## Ferramentas de Sistema (Nível 1)

Além de cálculo e data/hora, o agente agora suporta ações locais no projeto:

- `listar pasta <caminho>`
- `ler arquivo <caminho>`
- `criar arquivo <caminho> ::: <conteudo>`
- `executar comando <comando>`

Exemplos:

```text
listar pasta src/
ler arquivo README.md
criar arquivo notas/planejamento.md ::: objetivo: melhorar RAG
executar comando ls -la src
```

Observações:
- As operações de arquivo ficam restritas à pasta do projeto.
- Comandos potencialmente destrutivos são bloqueados por segurança.

## Consumo de RAM por Cenário

### Perfil ultra-leve (LFM2.5)

| Cenário | RAM | Tempo |
|---------|-----|-------|
| Nível 1: ferramenta/cache | ~50 MB | <10ms |
| Nível 1: ChromaDB busca | ~300 MB | ~50ms |
| Nível 2: LFM2.5-1.2B | ~731 MB | ~100-500ms |
| Nível 3: LFM2.5-8B-A1B MoE | ~1.5 GB | 1-4s |
| Nível 3: Coder 3B (código) | ~2.5 GB | 2-5s |

### Perfil máximo (Qwen)

| Cenário | RAM | Tempo |
|---------|-----|-------|
| Nível 2: Qwen 1.7B | ~1.5 GB | ~200ms-1s |
| Nível 3: Qwen 4B + RAG | ~3.5 GB | 3-8s |
| Nível 3: Coder 3B + RAG | ~2.8 GB | 2-5s |

## Módulos

### `src/core/` — Núcleo
- **config.py**: Perfis de modelos, configuração de agentes, níveis e thresholds
- **llm.py**: Interface com Ollama (streaming, métricas, fallback)
- **classificador.py**: Heurísticas para decidir nível 1/2/3

### `src/agentes/` — Lógica dos agentes
- **coordenador.py**: Roteia pergunta para o agente certo (keywords → LLM fallback)
- **executor.py**: Pipeline de execução com os 3 níveis

### `src/memoria/` — 3 Camadas de memória
- **cache.py**: Hash exato → resposta imediata (JSON)
- **semantica.py**: ChromaDB → busca por similaridade vetorial
- **sqlite.py**: Histórico, resumos, contexto e métricas

### `src/ferramentas/` — Execução pré-LLM
- **resolver.py**: Cálculos matemáticos, data/hora
- **web.py**: Pesquisa DuckDuckGo

## Personalização

### Trocar perfil de modelos

```python
# src/core/config.py
PERFIL_ATIVO = "equilibrado"  # ou "ultra_leve" ou "maximo"
```

### Adicionar agente

```python
# src/core/config.py
AGENTES["devops"] = {
    "modelo_rapido": MODELOS["rapido"],
    "modelo_profundo": MODELOS["completo"],
    "system_prompt": "Você é especialista DevOps...",
    "palavras_chave": ["docker", "k8s", "ci/cd", "pipeline"],
    "nivel_preferido": 3,
}
```

### Ajustar thresholds

```python
# src/core/config.py
CHROMADB_THRESHOLD = 0.62       # Filtro semântico inicial (recall)
CHROMADB_NIVEL1_THRESHOLD = 0.86 # Confiança para resposta direta no Nível 1
CHROMADB_TOP_K = 6               # Candidatos para re-ranking híbrido
RAG_MAX_DOCS = 4                 # Quantos docs entram no prompt final
RAG_MAX_CHARS = 2200             # Tamanho máximo de contexto RAG
```

Com isso, o fluxo fica mais preciso em hardware limitado:
- Busca semântica ampla (top_k maior) para não perder contexto.
- Re-ranking híbrido (semântico + lexical) para reduzir falso positivo.
- Contexto RAG compacto para evitar "diluição" da resposta em modelo pequeno.

### Criar perfil customizado

```python
# src/core/config.py
PERFIS["meu_perfil"] = {
    "coordenador": "LiquidAI/lfm2.5-1.2b-instruct",
    "rapido": "LiquidAI/lfm2.5-1.2b-instruct",
    "coder": "deepseek-coder:1.3b",      # Alternativa ultra-leve para código
    "completo": "maternion/lfm2.5",
    "embedding": "nomic-embed-text",
}
```
