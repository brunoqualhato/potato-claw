# Design: Hub de conexões, extensibilidade e usabilidade (fork local-first do potato-claw)

- **Data:** 2026-06-23
- **Autor:** Nikolas de Hor
- **Tipo:** Design / spec de evolução arquitetural
- **Status:** Aprovado (brainstorming), pronto para virar plano de implementação

## 1. Contexto

O `potato-claw` (fork em `nikolasdehor/potato-claw`, upstream `brunoqualhato/potato-claw`) é um framework multiagente em Python sobre Ollama, focado em rodar em "computador de batata" (8 GB, sem GPU, offline, sem API, processamento 100% local). Ele já tem um diferencial forte: multiagente real (coordenador/executor), RAG semântico, roteamento em 3 níveis e cache em 3 camadas.

Este design evolui o projeto como **fork próprio do Nikolas** (sem a restrição de "PR aceitável pro Bruno"), pegando ideias dos projetos irmãos que o Nikolas já ajudou:

- **nanobot** (`HKUDS/nanobot`, Python, ~4.000 linhas): ancestral do picoclaw. Por ser Python, é fonte de **cópia quase direta**.
- **picoclaw** (`sipeed/picoclaw`, Go): reescrita madura do nanobot, com hub de canais robusto. Fonte de **adaptação/inspiração** (Go para Python).

Genealogia: OpenClaw (430k linhas) -> nanobot (Python, enxuto) -> picoclaw (Go). O potato-claw é primo do nanobot.

## 2. Objetivo

Adicionar ao potato-claw um **hub de conexões completo** mais melhorias de memória de sessão e usabilidade, nas 4 frentes pedidas:

1. Canais de mensageria (Telegram primeiro; arquitetura para Discord/WhatsApp/etc depois).
2. Hub de ferramentas via MCP.
3. Skills plugáveis (pasta/markdown) com skill-creator.
4. Abstração de provider de LLM (mantendo Ollama como padrão).

Tudo sem perder o que torna o potato-claw bom.

## 3. Princípios (invioláveis)

1. **Offline-first, online-opcional.** O LLM roda sempre local (Ollama). Rede é apenas I/O opcional (canais externos, baixar skills) e **desligável por config**. Sem config de canais, o comportamento é o atual: CLI 100% offline.
2. **Preservar o diferencial.** Multiagente (coordenador/executor), RAG semântico, roteamento em 3 níveis e cache em 3 camadas ficam intactos. Não copiamos a "memória leve" do nanobot porque, em RAG, o potato-claw já está à frente; copiar seria regressão.
3. **PC fraco acima de tudo.** Toda camada nova é lazy/opt-in. Nada residente competindo RAM (mesma filosofia do PR #1: keep_alive efêmero, KV cache com teto). Dependências novas pesadas entram como extras opcionais, nunca no caminho padrão.
4. **Camadas aditivas.** O core atual (`src/agentes`, `src/core`, `src/memoria`, `src/ferramentas`) não é reorganizado. As capacidades novas entram como camadas em volta.

## 4. Estado atual (confirmado no repo)

```
api.py / main.py                 entradas (API FastAPI + CLI)
src/
  agentes/   base, coordenador, executor, sessao_codigo, templates   (multiagente)
  core/      analisador, classificador, config, llm(Ollama), utils    (roteamento 3 niveis)
  ferramentas/ resolver, web, web_async, web_rag                      (tools locais)
  memoria/   cache(3 camadas), semantica(RAG), sqlite                 (memoria forte)
tests/       66+ testes (sem CI no repo)
```

Lacunas frente a nanobot/picoclaw: nenhum canal externo, nenhum message bus, LLM (Ollama) acoplado em `core/llm.py`, sem MCP, sem sistema de skills, sem cron/heartbeat, sem gestão de sessão por usuário/canal.

## 5. Arquitetura alvo (camadas aditivas)

```
src/
  agentes/      INTACTO  (coordenador, executor, sessao_codigo)
  core/         + passa a chamar provedores/ no lugar do Ollama hardcoded
  memoria/      INTACTO  (cache, rag, sqlite)
  ferramentas/  + recebe tools expostas via MCP
  conexoes/     NOVO     bus.py + channels/ + manager.py
  provedores/   NOVO     base(ABC) + ollama(default) + litellm(opcional) + registry
  extensoes/    NOVO     mcp/ + skills/
  sessao/       NOVO     gestor de sessao por canal/usuario (anti-poisoning, trimming)
main.py / api.py   seguem como entradas equivalentes (viram "canais" no bus)
```

### Fluxo de dados

```
canal (Telegram/CLI/API)
   -> conexoes/bus  (InboundMessage)
   -> agentes/coordenador  (roteamento 3 niveis: ferramenta local -> modelo rapido -> modelo profundo + RAG)
   -> executor / ferramentas (locais + MCP) / memoria (RAG)
   -> conexoes/bus  (OutboundMessage)
   -> conexoes/manager  (retry/backoff/split + allow-list)
   -> canal de origem
```

CLI e API deixam de ser caminhos especiais: viram apenas dois canais entre os demais, publicando e consumindo do mesmo bus. Isso unifica o tratamento e habilita proatividade (cron/heartbeat publicando no bus sem um humano na ponta).

## 6. Componentes novos (detalhe)

### 6.1 `provedores/` (abstração de LLM)

- `base.py`: `LLMProvider` (ABC) com `complete()`/`stream()` e dataclass de resposta (texto, tool_calls, usage). Espelha `nanobot/providers/base.py:32`.
- `ollama.py`: `OllamaProvider`, provider **padrão**. Move a lógica de `src/core/llm.py` para cá, preservando o tuning do PR #1 (num_ctx por nível, num_thread, keep_alive efêmero, timeout).
- `litellm.py`: `LiteLLMProvider`, **opcional e desligado por padrão** (extra `[litellm]` no pyproject). Dá 100+ backends, incluindo cloud, como escape hatch consciente. Espelha `nanobot/providers/litellm_provider.py:25`.
- `registry.py`: registro nome -> factory (espelha `nanobot/providers/registry.py`). `core/config.py` escolhe o provider; default = ollama.
- `core/llm.py` passa a delegar ao provider resolvido pelo registry. Os agentes não mudam.

### 6.2 `conexoes/bus.py` (message bus)

- Tipos `InboundMessage`, `OutboundMessage`, `SenderInfo` (espelha `picoclaw/pkg/bus/types.go`).
- `MessageBus` async (pub/sub) desacoplando canais do coordenador. Implementação leve (asyncio queues), sem broker externo.

### 6.3 `conexoes/channels/` (hub de canais)

- `base.py`: `BaseChannel` (ABC) com `start()`/`stop()`/`send()` async + `is_allowed_sender()` (allow-list). Espelha `nanobot/channels/base.py:12` e o `Channel` interface do `picoclaw/pkg/channels/base.go:43`.
- **Capabilities opcionais** via duck typing / `typing.Protocol` (typing, edit, reaction, placeholder), no padrão de interface-segregation do `picoclaw/pkg/channels/interfaces.go`.
- `manager.py`: `ChannelManager` com **retry/backoff exponencial**, split de mensagens longas e classificação de erro de envio. Espelha `picoclaw/pkg/channels/manager.go` (SendWithRetry, preSend) e `errutil.go`.
- `registry.py`: registro de factories de canal, auto-registro por módulo (padrão `RegisterFactory` do `picoclaw/pkg/channels/registry.go`).
- Canais concretos:
  - `cli.py`: o CLI atual vira um canal no bus (prova o hub sem rede).
  - `telegram.py`: **1º canal externo**, via long-polling (sem webhook público). Espelha `nanobot/channels/telegram.py`.

### 6.4 `extensoes/skills/` (skills plugáveis)

- Loader de skills em pasta/markdown (frontmatter + corpo), no modelo do `nanobot/skills/` e `nanobot/agent/skills.py:101` (`build_skills_summary`).
- Skills iniciais portadas: `summarize`, `memory` (integradas ao RAG existente, não substituindo).
- `skill-creator`: skill que cria skills (espelha `nanobot/skills/skill-creator`).
- Descoberta remota (estilo ClawHub) fica **opt-in e online** (não no caminho padrão offline).

### 6.5 `extensoes/mcp/` (hub de ferramentas)

- Cliente MCP que conecta servidores MCP **locais** (offline-friendly) ou remotos (opt-in).
- Tools MCP são expostas ao agente via `src/ferramentas/resolver.py` (ponto de integração já existente), sem mexer no coordenador.

### 6.6 `sessao/` (memória de sessão por canal/usuário)

- `Session` + `SessionManager` por (canal, usuário), com **isolamento anti-poisoning** e **trimming/summarize** de janela de contexto. Espelha `nanobot/session/manager.py:16` e `nanobot/agent/context.py:15` (`ContextBuilder`).
- Complementa o RAG (longo prazo) com contexto de conversa (curto prazo). Não substitui `src/memoria/`.

### 6.7 Proatividade e UX

- `cron`/`heartbeat`: tarefas agendadas e agente proativo publicando no bus (espelha `nanobot/cron`, `nanobot/heartbeat`). Local, offline-friendly.
- Launcher/TUI de onboarding e config, inspirado em `picoclaw/cmd/picoclaw-launcher-tui`. Reduz fricção de setup (escolher perfil de modelo, ligar/desligar canais).

## 7. Roadmap em 4 fases

Priorizado por dependência técnica, impacto e esforço. Cada fase é um incremento testável e isolado.

### Fase 1 - Fundação (risco baixo, habilitadores)
- Sincronizar fork com upstream do Bruno (2 commits atrás).
- `provedores/`: `LLMProvider` ABC + `OllamaProvider` (default, com tuning do PR #1) + `registry`. `litellm` como extra opcional desligado. `core/llm.py` passa a delegar.
- `conexoes/bus.py`: tipos + `MessageBus` async.
- **CI** (GitHub Actions: pytest + lint) - o repo não tem.
- Gate: 66+ testes atuais continuam verdes.

### Fase 2 - Hub de canais (maior impacto de uso)
- `conexoes/channels/`: `BaseChannel` ABC + `ChannelManager` (retry/split/allow-list) + `registry`.
- CLI vira canal no bus.
- `telegram.py` (long-polling) como 1º canal externo.
- Coordenador passa a consumir do bus.

### Fase 3 - Extensibilidade (skills + MCP)
- `extensoes/skills/`: loader pasta/markdown + skills `summarize`/`memory` + `skill-creator`.
- `extensoes/mcp/`: cliente MCP integrado a `ferramentas/resolver.py`.

### Fase 4 - Sessão, proatividade e UX
- `sessao/`: `SessionManager` por canal/usuário (anti-poisoning, trimming).
- `cron`/`heartbeat`.
- Launcher/TUI de onboarding.

## 8. Decisões tomadas

- **Nome:** mantém `potato-claw`.
- **1º canal externo:** Telegram (long-polling).
- **litellm:** incluído como provider opcional, desligado por padrão (extra `[litellm]`).
- **Arquitetura:** camadas aditivas (Abordagem A), core intacto.
- **Estratégia:** fork próprio do Nikolas; sem dependência do aval do Bruno.

## 9. Tratamento de erro e offline

- `conexoes` e `extensoes` são **opt-in via config**. Sem config = CLI offline puro (comportamento atual).
- Falha de rede/canal **não derruba o core**. `ChannelManager` com retry/backoff exponencial e degradação graciosa (padrão do picoclaw).
- Provider: Ollama indisponível gera erro claro. **Sem fallback cloud automático** (respeita offline-first); cloud só se o usuário ligar litellm explicitamente.
- MCP/skills remotos: erro de rede é não-fatal; o agente segue com ferramentas/skills locais.

## 10. Estratégia de testes

- **Não-regressão é gate:** os 66+ testes atuais continuam verdes em toda fase.
- Unit tests por camada nova:
  - bus: pub/sub, ordering, backpressure.
  - channel manager: retry/backoff, split, allow-list (com canal mock, padrão de `picoclaw/pkg/channels/manager_test.go`).
  - provider registry: resolução e default ollama.
  - skill loader: parsing de frontmatter, summary.
  - mcp client: handshake e exposição de tools (com servidor MCP local de teste).
- CI roda pytest + lint em cada push (introduzido na Fase 1).
- Cobertura alvo: manter/elevar a baseline atual; mínimo 80% nas camadas novas.

## 11. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Camadas novas estourarem RAM no PC fraco | Lazy/opt-in; nada residente; medir RAM antes/depois (mesma disciplina do PR #1) |
| "Offline puro" comprometido por canais | Tudo desligável; default sem rede; doc explícita |
| Misturar filosofias (single vs multiagente) | Manter coordenador/executor como cérebro; bus só transporta |
| litellm puxar peso/cloud sem querer | Extra opcional, desligado por padrão, sem import no caminho default |
| Divergência do upstream do Bruno | Sincronizar na Fase 1; manter core compatível para cherry-pick eventual |

## 12. Fora de escopo (YAGNI por enquanto)

- Webhooks públicos / deploy exposto (Telegram via long-polling resolve sem isso).
- Discord/WhatsApp/Matrix no primeiro ciclo (arquitetura preparada, implementação depois).
- Descoberta/instalação remota de skills (ClawHub) no caminho padrão.
- Voz/mídia avançada (picoclaw tem `voice`/`media`; adiar).
- Mission Control / UI web (não é o foco local-first).
