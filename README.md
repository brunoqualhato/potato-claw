# 🥔 Potato-Claw

[![CI](https://github.com/brunoqualhato/potato-claw/actions/workflows/ci.yml/badge.svg)](https://github.com/brunoqualhato/potato-claw/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-purple.svg)](https://ollama.com/)

Sistema multiagente local em Python com execução em **3 níveis de performance**, memória persistente e pesquisa web com grounding.
Inspirado na ideia de agentes inteligentes como o OpenClaw, mas projetado para rodar em **batatas** — Mac M1 com 8GB de RAM, laptops modestos, hardware limitado.

> *"Se o OpenClaw é para quem tem GPU de sobra, o Potato-Claw é para quem roda tudo no processador de micro-ondas."*

## O que o projeto faz

O Potato-Claw recebe uma pergunta, analisa a intenção com uma LLM coordenadora e decide entre:

- responder com **ferramentas locais** sem chamar LLM;
- usar um **modelo rápido** com contexto curto;
- usar um **modelo profundo** com RAG local e/ou pesquisa web.

Além do modo de chat, o projeto também inclui:

- **CLI interativa e batch** (`main.py`);
- **API HTTP com FastAPI** (`api.py`);
- **memória em 3 camadas**: cache JSON, SQLite e ChromaDB;
- **loop autônomo de geração de código** via `/projeto`;
- **web RAG** com busca, fetch, limpeza de HTML e extração de fatos;
- **template library** para geração de projetos estruturados;
- **self-correction loop** para maximizar qualidade com modelos pequenos.

## Arquitetura resumida

```text
Pergunta do usuário
        │
        ▼
Analisador de intenção (format=json, few-shot dinâmico)
        │
        ├── ferramenta local → src/ferramentas/resolver.py
        ├── nível 2 rápido   → modelo leve + contexto curto + self-correction
        └── nível 3 profundo → web RAG + ChromaDB + modelo profundo
```

O pipeline principal vive em `src/agentes/executor.py` e orquestra:

1. análise de intenção (com few-shot dinâmico via ChromaDB);
2. tentativa de resolução por ferramenta;
3. cache exato;
4. recuperação semântica via ChromaDB;
5. self-correction antes de promover nível;
6. chamada ao LLM rápido ou profundo.

## Estrutura do projeto

```text
.
├── api.py                      # API HTTP mínima com FastAPI + auth
├── main.py                     # CLI interativa, batch e comandos /projeto
├── requirements.txt            # Dependências principais
├── requirements-dev.txt        # Dependências de teste/lint
├── pytest.ini
├── .env.example                # Variáveis de ambiente suportadas
├── data/                       # Persistência local (criado em runtime)
├── src/
│   ├── agentes/
│   │   ├── base.py             # Classe base e factory de agentes
│   │   ├── coordenador.py      # Validação de prompt + roteamento por keywords
│   │   ├── executor.py         # Pipeline principal (3 níveis + self-correction)
│   │   ├── sessao_codigo.py    # Loop autônomo para construção de projetos
│   │   └── templates.py        # Template library para projetos estruturados
│   ├── core/
│   │   ├── analisador.py       # Classificador semântico (format=json + few-shot)
│   │   ├── classificador.py    # Classificação de complexidade por nível
│   │   ├── config.py           # Perfis, modelos, thresholds e env
│   │   ├── llm.py              # Integração com Ollama (stream + degeneration abort)
│   │   ├── logging_config.py   # Logging centralizado
│   │   └── utils.py            # Utilitários simples
│   ├── ferramentas/
│   │   ├── resolver.py         # Cálculo, data/hora, arquivos e comandos locais
│   │   ├── web.py              # Busca simples com DuckDuckGo
│   │   ├── web_async.py        # Paralelização leve para tarefas I/O bound
│   │   └── web_rag.py          # Search → Fetch → Convert → Extract
│   └── memoria/
│       ├── cache.py            # Cache JSON com LRU + escrita atômica
│       ├── semantica.py        # Memória vetorial com ChromaDB + logging
│       └── sqlite.py           # Histórico, resumos e métricas (WAL mode)
└── tests/
    ├── conftest.py             # Fixtures de isolamento (singleton reset)
    ├── test_analisador.py
    ├── test_cache.py
    ├── test_classificador.py
    ├── test_coordenador.py
    ├── test_executor.py        # Testes de integração do pipeline
    ├── test_ferramentas.py
    ├── test_sessao_codigo.py   # Testes de parsing, validação, snapshot
    ├── test_utils.py
    └── test_web_rag.py
```

## Perfis de modelos

A configuração fica em `src/core/config.py` e também pode ser controlada por `.env` com `NEURON_PERFIL`.

| Perfil | Coordenador | Rápido | Profundo | Código | Embedding |
|---|---|---|---|---|---|
| `ultra_leve` | LFM2.5 1.2B | LFM2.5 1.2B | maternion/lfm2.5 | qwen2.5-coder:3b | nomic-embed-text |
| `equilibrado` | LFM2.5 1.2B | qwen3:1.7b | qwen3:4b | qwen2.5-coder:3b | nomic-embed-text |
| `maximo` | qwen3:1.7b | qwen3:1.7b | qwen3:4b | qwen2.5-coder:3b | nomic-embed-text |

## Instalação

### 1. Instale e suba o Ollama

```bash
brew install ollama
ollama serve
```

### 2. Baixe os modelos do perfil desejado

Exemplo para o perfil padrão `ultra_leve`:

```bash
ollama pull LiquidAI/lfm2.5-1.2b-instruct
ollama pull maternion/lfm2.5
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text
```

### 3. Instale dependências Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para desenvolvimento:

```bash
pip install -r requirements-dev.txt
```

### 4. Configure ambiente

```bash
cp .env.example .env
```

Variáveis importantes:

- `NEURON_PERFIL` — perfil de modelos (ultra_leve/equilibrado/maximo)
- `NEURON_API_KEY` — autenticação da API HTTP (vazio = acesso livre)
- `NEURON_LOG_LEVEL` — nível de log (DEBUG/INFO/WARNING/ERROR)
- `NEURON_DEBUG` — atalho para DEBUG (true/false)

## Como executar

### CLI interativa

```bash
python main.py
```

Com debug:

```bash
python main.py --debug
```

### Modo batch

```bash
python main.py --query "como usar docker compose"
python main.py --json --query "qual a última versão do node.js"
python main.py --agente programador --nivel 3 --query "crie uma API REST com JWT"
```

### API HTTP

```bash
pip install fastapi uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000
```

Com autenticação (defina `NEURON_API_KEY` no `.env`):

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sua-chave" \
  -d '{"pergunta":"quanto é 15% de 300"}'
```

## Comandos da CLI

| Comando | Descrição |
|---|---|
| `/nivel <1\|2\|3>` | Força o nível da próxima execução |
| `/agente <nome>` | Força o agente |
| `/stats` | Exibe métricas de performance |
| `/knowledge <texto>` | Adiciona conhecimento ao ChromaDB |
| `/ingest <arquivo>` | Ingestão de arquivo texto em chunks |
| `/feedback <texto>` | Salva preferência para aprendizado futuro |
| `/contexto` | Mostra histórico recente |
| `/resumo` | Mostra o último resumo da conversa |
| `/limpar` | Limpa histórico |
| `/limparcache` | Limpa o cache exato |
| `/modelos` | Mostra os modelos configurados |
| `/projeto <descrição>` | Executa loop autônomo de geração de projeto |
| `/projeto -i <descrição>` | Modo interativo (feedback a cada step) |
| `/continuar` | Avança o próximo step manualmente |
| `/rerun <N>` | Reexecuta um step do projeto |
| `/editar <arquivo> <instrução>` | Edita arquivo da sessão |
| `/exportar` | Exporta os arquivos gerados |
| `/gc` | Garbage collection do cache vetorial |
| `/qualidade` | Métricas da sessão de código |
| `/ajuda` | Lista de comandos |
| `/sair` | Encerrar |

## Otimizações para Hardware Fraco

O Potato-Claw implementa várias técnicas para extrair máxima qualidade de modelos 1.2B–4B:

- **`format="json"`** — força output JSON estruturado direto do Ollama
- **Few-shot dinâmico** — seleciona exemplos relevantes do histórico via ChromaDB
- **Self-correction loop** — tenta corrigir respostas fracas antes de promover nível
- **Template library** — esqueletos pré-definidos reduzem carga cognitiva do modelo
- **Warm-up keep-alive** — pré-carrega modelos na RAM ao iniciar
- **Stream abort** — detecta degeneração (repetição) e para imediatamente
- **Feedback → memória** — preferências do usuário melhoram respostas futuras
- **Prompts compactos** — ~300 tokens liberando espaço para contexto real

## Testes

```bash
pytest
```

Cobertura:
- Pipeline principal (integração com mocks do Ollama)
- Parsing do analisador de intenção
- Classificador de complexidade
- Ferramentas locais (cálculo, data/hora, arquivos)
- Sessão de código (planejamento, validação, snapshot/rollback)
- Cache e memória
- Conversão HTML → Markdown no web RAG

## Licença

MIT
