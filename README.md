# 🧠 Neuron

Sistema multiagente local em Python com execução em **3 níveis de performance**, memória persistente e pesquisa web com grounding.
Ele foi pensado para rodar com **Ollama local** em máquinas modestas, incluindo **Mac M1 com 8 GB de RAM**, priorizando respostas baratas quando possível e promovendo para pipelines mais profundos apenas quando necessário.

## O que o projeto faz

O Neuron recebe uma pergunta, analisa a intenção com uma LLM coordenadora e decide entre:

- responder com **ferramentas locais** sem chamar LLM;
- usar um **modelo rápido** com contexto curto;
- usar um **modelo profundo** com RAG local e/ou pesquisa web.

Além do modo de chat, o projeto também inclui:

- **CLI interativa e batch** (`main.py`);
- **API HTTP com FastAPI** (`api.py`);
- **memória em 3 camadas**: cache JSON, SQLite e ChromaDB;
- **loop autônomo de geração de código** via `/projeto`;
- **web RAG** com busca, fetch, limpeza de HTML e extração de fatos.

## Arquitetura resumida

```text
Pergunta do usuário
        │
        ▼
Analisador de intenção (src/core/analisador.py)
        │
        ├── ferramenta local → src/ferramentas/resolver.py
        ├── nível 2 rápido   → modelo leve + contexto curto
        └── nível 3 profundo → web RAG + ChromaDB + modelo profundo
```

O pipeline principal vive em `src/agentes/executor.py` e orquestra:

1. análise de intenção;
2. tentativa de resolução por ferramenta;
3. cache exato;
4. recuperação semântica via ChromaDB;
5. chamada ao LLM rápido ou profundo.

## Estrutura do projeto

```text
.
├── api.py                      # API HTTP mínima com FastAPI
├── main.py                     # CLI interativa, batch e comandos /projeto
├── requirements.txt            # Dependências principais
├── requirements-dev.txt        # Dependências de teste/lint
├── pytest.ini
├── .env.example                # Variáveis de ambiente suportadas
├── data/                       # Persistência local (criado/uso em runtime)
├── src/
│   ├── agentes/
│   │   ├── base.py             # Classe base e factory de agentes
│   │   ├── coordenador.py      # Roteamento/validação por heurística + LLM
│   │   ├── executor.py         # Pipeline principal de execução
│   │   └── sessao_codigo.py    # Loop autônomo para construção de projetos
│   ├── core/
│   │   ├── analisador.py       # Classificador semântico de intenção em JSON
│   │   ├── classificador.py    # Classificação de complexidade por nível
│   │   ├── config.py           # Perfis, modelos, thresholds e env
│   │   ├── llm.py              # Integração com Ollama
│   │   └── utils.py            # Utilitários simples
│   ├── ferramentas/
│   │   ├── resolver.py         # Cálculo, data/hora, arquivos e comandos locais
│   │   ├── web.py              # Busca simples com DuckDuckGo
│   │   ├── web_async.py        # Paralelização leve para tarefas I/O bound
│   │   └── web_rag.py          # Search → Fetch → Convert → Extract
│   └── memoria/
│       ├── cache.py            # Cache JSON com LRU
│       ├── semantica.py        # Memória vetorial com ChromaDB
│       └── sqlite.py           # Histórico, resumos e métricas
└── tests/
    ├── test_analisador.py
    ├── test_cache.py
    ├── test_classificador.py
    ├── test_coordenador.py
    ├── test_ferramentas.py
    ├── test_utils.py
    └── test_web_rag.py
```

## Perfis de modelos

A configuração fica em `src/core/config.py` e também pode ser controlada por `.env` com `NEURON_PERFIL`.

Perfis disponíveis:

- `ultra_leve`
- `equilibrado`
- `maximo`

Resumo do mapeamento atual:

| Perfil | Coordenador | Rápido | Profundo | Código | Embedding |
|---|---|---|---|---|---|
| `ultra_leve` | `LiquidAI/lfm2.5-1.2b-instruct` | `LiquidAI/lfm2.5-1.2b-instruct` | `maternion/lfm2.5` | `qwen2.5-coder:3b` | `nomic-embed-text` |
| `equilibrado` | `LiquidAI/lfm2.5-1.2b-instruct` | `qwen3:1.7b` | `qwen3:4b` | `qwen2.5-coder:3b` | `nomic-embed-text` |
| `maximo` | `qwen3:1.7b` | `qwen3:1.7b` | `qwen3:4b` | `qwen2.5-coder:3b` | `nomic-embed-text` |

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

Se quiser usar API HTTP, também instale as dependências opcionais comentadas em `requirements.txt`:

```bash
pip install fastapi uvicorn
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

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Variáveis importantes:

- `NEURON_PERFIL`
- `NEURON_DATA_DIR`
- `NEURON_CACHE_HABILITADO`
- `NEURON_CHROMADB_TOP_K`
- `NEURON_CHROMADB_THRESHOLD`
- `NEURON_CHROMADB_NIVEL1_THRESHOLD`
- `NEURON_RAG_MAX_DOCS`
- `NEURON_RAG_MAX_CHARS`
- `NEURON_WEB_RAG_MAX_PAGINAS`
- `NEURON_WEB_RAG_FETCH_TIMEOUT`
- `NEURON_WEB_RAG_MAX_MD_CHARS`
- `NEURON_WEB_RAG_CACHE_TTL`
- `NEURON_CONTEXTO_MAX_MSGS`
- `OLLAMA_HOST` (opcional, para host remoto)

## Como executar

### CLI interativa

```bash
python main.py
```

### Modo batch

```bash
python main.py --query "como usar docker compose"
python main.py --json --query "qual a última versão do node.js"
python main.py --agente programador --nivel 3 --query "crie uma API REST com JWT"
```

### API HTTP

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints disponíveis:

- `GET /health`
- `GET /stats`
- `POST /chat`
- `POST /chat/agente`

Exemplo:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pergunta":"quanto é 15% de 300"}'
```

## Comandos da CLI

A interface expõe comandos de operação e automação:

| Comando | Descrição |
|---|---|
| `/nivel <1|2|3>` | Força o nível da próxima execução |
| `/agente <nome>` | Força o agente (`generalista`, `programador`, `pesquisador`, `analista`) |
| `/stats` | Exibe métricas de performance |
| `/knowledge <texto>` | Adiciona conhecimento ao ChromaDB |
| `/ingest <arquivo>` | Ingestão de arquivo texto em chunks |
| `/contexto` | Mostra histórico recente |
| `/resumo` | Mostra o último resumo da conversa |
| `/limpar` | Limpa histórico |
| `/limparcache` | Limpa o cache exato |
| `/modelos` | Mostra os modelos configurados |
| `/projeto <descrição>` | Executa loop autônomo de geração de projeto |
| `/projeto -i <descrição>` | Executa o loop em modo interativo |
| `/continuar` | Avança o próximo step manualmente |
| `/rerun <N>` | Reexecuta um step do projeto |
| `/editar <arquivo> <instrução>` | Edita arquivo da sessão de projeto |
| `/exportar` | Exporta os arquivos gerados |
| `/gc` | Executa garbage collection do cache vetorial |
| `/qualidade` | Exibe métricas da sessão de código |
| `/ajuda` | Mostra a lista de comandos |
| `/sair` | Encerra o programa |

## Ferramentas locais do nível 1

O nível 1 tenta resolver tarefas sem custo de LLM quando possível.

Capacidades implementadas em `src/ferramentas/resolver.py`:

- cálculo matemático com AST seguro;
- data e hora local ou por fuso conhecido;
- leitura/criação/listagem de arquivos restrita à pasta do projeto;
- execução de comandos locais com **allowlist**.

Exemplos de uso:

```text
listar pasta src/
ler arquivo README.md
criar arquivo notas/planejamento.md ::: objetivo: melhorar o RAG
executar comando ls -la src
```

Alguns comandos permitidos:

- `ls`, `cat`, `head`, `tail`, `wc`, `echo`, `pwd`, `find`, `grep`
- `tree`, `file`, `du`, `df`, `which`, `whoami`, `date`, `uname`
- `pip`, `pip3`, `node`, `npm`, `git`, `jq`, `sed`, `awk`, `sort`, `uniq`, `diff`

Há bloqueios explícitos para ações destrutivas, incluindo subcomandos como `git push`, `git remote` e `git config`, além de flags como `--force`, `-rf`, `--hard` e `--no-preserve-root`.

## Memória em 3 camadas

### 1. Cache exato (`src/memoria/cache.py`)

- persistido em JSON;
- política LRU;
- até `500` entradas;
- evita cachear consultas genéricas como `oi`.

### 2. Memória semântica (`src/memoria/semantica.py`)

- ChromaDB persistente em `data/chromadb`;
- embeddings locais via Ollama (`nomic-embed-text`);
- re-ranking híbrido com sinal semântico + lexical.

### 3. Memória relacional (`src/memoria/sqlite.py`)

- SQLite em `data/memoria.db`;
- armazena histórico, contexto, resumos e métricas.

## Web RAG

O módulo `src/ferramentas/web_rag.py` implementa um pipeline mais robusto que uma busca por snippets:

1. **Search** com DuckDuckGo (`ddgs`);
2. **Fetch** das páginas reais;
3. **Convert** de HTML para Markdown limpo;
4. **Extract** com uma LLM leve para manter apenas fatos relevantes;
5. uso desse contexto no prompt final.

Há também uma versão rápida para nível 2 e uma versão profunda para nível 3.

## Testes

Executar a suíte:

```bash
pytest
```

Os testes atuais cobrem principalmente:

- parsing do analisador de intenção;
- classificador de complexidade;
- roteamento por palavras-chave;
- cache;
- ferramentas locais;
- conversão HTML → Markdown no web RAG.

## Observações importantes

- O `README` antigo descrevia parte da arquitetura, mas o código atual já evoluiu para incluir `src/core/analisador.py`, `src/agentes/sessao_codigo.py`, `src/ferramentas/web_async.py` e `src/ferramentas/web_rag.py`.
- `api.py` usa `FastAPI`, `Pydantic` e `uvicorn`, mas essas dependências ainda estão comentadas em `requirements.txt`, então precisam ser instaladas separadamente para a API funcionar.
- A pasta `data/` no repositório contém apenas `.gitkeep`; os arquivos reais (`cache.json`, `memoria.db`, `chromadb/`, `web_cache/`) são gerados durante o uso.

## Próximos passos sugeridos

- adicionar instruções de instalação separadas para uso somente CLI vs uso com API;
- documentar melhor o fluxo do `/projeto` e o formato dos artefatos exportados;
- incluir exemplos de request/response da API e cenários de uso por agente.
