"""
Motor de execução dos agentes com 3 níveis de performance.

Pipeline por nível:
  Nível 1: Ferramentas → Cache → ChromaDB → RETORNA (sem LLM!)
  Nível 2: ...nivel1... → Modelo Rápido (1.7B) + contexto mínimo
  Nível 3: ...nivel1... → ChromaDB RAG → Modelo Completo (4B) + contexto rico
"""

import time
from rich.console import Console
from rich.panel import Panel

from src.core.config import (
    AGENTES,
    NIVEIS,
    CHROMADB_NIVEL1_THRESHOLD,
    RAG_MAX_CHARS,
    RAG_MAX_DOCS,
    NUM_CTX_NIVEL,
    KEEP_ALIVE_PRINCIPAL,
)
from src.core.classificador import classificar_complexidade, explicar_nivel
from src.core.llm import chamar_llm, resumir_conversa, verificar_modelo_disponivel
from src.core.analisador import analisar_intencao, IntencaoAnalisada
from src.memoria.cache import Cache
from src.memoria.sqlite import Memoria
from src.memoria.semantica import MemoriaSemantica
from src.ferramentas.resolver import (
    executar_ferramentas, obter_data_hora, calcular,
    verificar_ferramenta_saudacao, _extrair_local_hora,
    listar_pasta, ler_arquivo, criar_arquivo, executar_comando_local,
)
from src.ferramentas.web_rag import pesquisar_web_profunda, pesquisar_web_rapida
from src.ferramentas.web_async import paralelo_sync
from src.agentes.base import Agente, ConfigAgente, criar_agente

console = Console()


class SistemaAgentes:
    """Gerencia a execução dos agentes com 3 níveis de performance."""

    def __init__(self):
        self.memoria = Memoria()
        self.cache = Cache()
        self.semantica = MemoriaSemantica()
        self.nivel_forcado: int | None = None
        self._agentes: dict[str, Agente] = self._inicializar_agentes()

    def _inicializar_agentes(self) -> dict[str, Agente]:
        """Cria instâncias de agentes a partir da config. Uma vez, no boot."""
        agentes = {}
        for nome, cfg in AGENTES.items():
            config = ConfigAgente(
                nome=nome,
                modelo_rapido=cfg["modelo_rapido"],
                modelo_profundo=cfg["modelo_profundo"],
                system_prompt=cfg["system_prompt"],
                palavras_chave=cfg.get("palavras_chave", []),
                nivel_preferido=cfg.get("nivel_preferido", 2),
                usa_web=cfg.get("usa_web", False),
            )
            agentes[nome] = criar_agente(nome, config)
        return agentes

    def _obter_agente(self, nome: str) -> Agente:
        """Retorna instância do agente (fallback para generalista)."""
        return self._agentes.get(nome, self._agentes["generalista"])

    def executar(self, nome_agente: str, pergunta: str) -> str:
        """
        Pipeline principal com análise de intenção via LLM.

        Fluxo:
          1. LLM analisa intenção (agente, web, ferramenta, params) — ~100ms
          2. Se ferramenta local resolve → retorna sem LLM (nível 1)
          3. Se precisa web → fetch + extract primeiro
          4. LLM final responde grounded nos dados coletados
        """
        agente_cfg = AGENTES[nome_agente]
        agente_obj = self._obter_agente(nome_agente)
        inicio = time.time()

        # ═══════════════════════════════════════════
        # ANÁLISE DE INTENÇÃO (LLM coordenadora — chamada ÚNICA)
        # ═══════════════════════════════════════════
        intencao = analisar_intencao(pergunta)
        console.print(
            f"[dim]🧠 Intenção: agente={intencao.agente}, web={intencao.precisa_web}, "
            f"ferramenta={intencao.ferramenta}[/dim]"
        )

        # Redireciona se analisador sugeriu agente diferente
        # (exceto se o agente foi forçado pelo usuário via /agente)
        if nome_agente == "generalista" and intencao.agente != "generalista":
            nome_agente = intencao.agente
            agente_cfg = AGENTES[nome_agente]
            agente_obj = self._obter_agente(nome_agente)
            console.print(f"[dim]🎯 {nome_agente}[/dim]")

        # Classificar complexidade
        if self.nivel_forcado:
            nivel = self.nivel_forcado
            self.nivel_forcado = None
        else:
            nivel = classificar_complexidade(pergunta)
            if nivel == 2 and agente_cfg.get("nivel_preferido", 2) == 3:
                if len(pergunta.split()) > 12:
                    nivel = 3

        # Hook pré-execução do agente (permite ajuste de nível/query)
        pergunta, nivel = agente_obj.pre_execucao(pergunta, nivel)

        console.print(f"[dim]{explicar_nivel(nivel)}[/dim]")

        # ═══════════════════════════════════════════
        # NÍVEL 1: FERRAMENTAS LOCAIS (sem LLM)
        # ═══════════════════════════════════════════

        # 1a. Ferramenta identificada pelo analisador
        resultado_ferramenta = self._executar_ferramenta_por_intencao(intencao, pergunta)
        if resultado_ferramenta:
            console.print(
                Panel(resultado_ferramenta, title="⚡ Nível 1 • Ferramenta", border_style="cyan")
            )
            self._salvar(pergunta, resultado_ferramenta, nome_agente, nivel=1, inicio=inicio)
            return resultado_ferramenta

        # 1b. Fallback: tenta resolver com ferramentas heurísticas (cálculo puro, expressão)
        resultado_heuristico = executar_ferramentas(pergunta)
        if resultado_heuristico:
            console.print(
                Panel(resultado_heuristico, title="⚡ Nível 1 • Ferramenta", border_style="cyan")
            )
            self._salvar(pergunta, resultado_heuristico, nome_agente, nivel=1, inicio=inicio)
            return resultado_heuristico

        # 1c. Cache exato — ignorado se precisa de dados frescos
        cache_key = f"{nome_agente}:{pergunta}"
        resposta_cache = None if intencao.precisa_web else self.cache.buscar(cache_key)
        if resposta_cache:
            console.print("[dim]📋 Nível 1 • Cache[/dim]")
            console.print(resposta_cache, style="green")
            self._salvar_metrica(nome_agente, 1, inicio, fonte="cache")
            return resposta_cache

        # 1d. ChromaDB — busca semântica
        docs_similares = self.semantica.buscar_similar(pergunta)
        if docs_similares and nivel == 1 and not intencao.precisa_web:
            melhor = docs_similares[0]
            if melhor.get("score_hibrido", melhor["similaridade"]) >= CHROMADB_NIVEL1_THRESHOLD:
                resposta = melhor["conteudo"].split("Resposta: ", 1)[-1]
                console.print(
                    f"[dim]🧲 Nível 1 • ChromaDB ({melhor.get('score_hibrido', melhor['similaridade']):.0%})[/dim]"
                )
                console.print(resposta, style="green")
                self._salvar_metrica(nome_agente, 1, inicio, fonte="chromadb")
                return resposta

        # Promover se nível 1 não resolveu
        if nivel == 1:
            nivel = 2
            console.print("[dim]↑ Promovido para Nível 2[/dim]")

        # ═══════════════════════════════════════════
        # NÍVEL 2: RÁPIDO (modelo 1.7B)
        # ═══════════════════════════════════════════

        if nivel == 2:
            contexto_busca = ""
            if intencao.precisa_web:
                console.print("[dim]🔍 Buscando na web (rápido)...[/dim]")
                contexto_busca = pesquisar_web_rapida(pergunta, max_paginas=2)

            mensagens = self._montar_contexto(2, contexto_busca, pergunta)

            resultado = chamar_llm(
                modelo=agente_cfg["modelo_rapido"],
                system_prompt=agente_cfg["system_prompt"],
                mensagens=mensagens,
                stream=True,
                max_tokens=512,
                temperatura=0.4,
                num_ctx=NUM_CTX_NIVEL[2],
                keep_alive=KEEP_ALIVE_PRINCIPAL,
            )

            self._salvar(pergunta, resultado["resposta"], nome_agente, nivel=2, inicio=inicio)
            self._salvar_metrica(
                nome_agente, 2, inicio,
                tokens_in=resultado["tokens_entrada"],
                tokens_out=resultado["tokens_saida"],
                fonte="llm_rapido",
            )
            if self._deve_promover_para_profundo(pergunta, resultado["resposta"]):
                console.print("[dim]↑ Ajuste de precisão: promovido para Nível 3[/dim]")
                nivel = 3
            else:
                resposta_final = agente_obj.pos_execucao(pergunta, resultado["resposta"])
                return resposta_final

        # ═══════════════════════════════════════════
        # NÍVEL 3: PROFUNDO (modelo 4B + RAG)
        # ═══════════════════════════════════════════

        # Paraleliza busca web + ChromaDB para reduzir latência
        contexto_busca = ""
        contexto_rag = ""

        tem_docs_chromadb = bool(docs_similares)

        if intencao.precisa_web and tem_docs_chromadb:
            console.print("[dim]🔍⚡ Web profunda + RAG em paralelo...[/dim]")
            resultados = paralelo_sync(
                (pesquisar_web_profunda, (pergunta, 3)),
                (self._construir_contexto_rag, (docs_similares[:RAG_MAX_DOCS],)),
            )
            contexto_busca = resultados[0] if not isinstance(resultados[0], Exception) else ""
            contexto_rag = resultados[1] if not isinstance(resultados[1], Exception) else ""
        else:
            if intencao.precisa_web:
                console.print("[dim]🔍 Busca web profunda (fetch + extract)...[/dim]")
                contexto_busca = pesquisar_web_profunda(pergunta, max_paginas=3)

            if tem_docs_chromadb:
                docs_rag = docs_similares[:RAG_MAX_DOCS]
                console.print(f"[dim]🧲 RAG: {len(docs_rag)} docs do ChromaDB[/dim]")
                contexto_rag = self._construir_contexto_rag(docs_rag)

        mensagens = self._montar_contexto(
            n_msgs=5,
            contexto_busca=contexto_busca,
            pergunta=pergunta,
            contexto_rag=contexto_rag,
        )

        modelo_profundo = agente_cfg["modelo_profundo"]
        if not verificar_modelo_disponivel(modelo_profundo):
            # Tenta encontrar o melhor modelo instalado antes de cair no rapido
            candidatos = ["qwen3:4b", "qwen2.5:3b", "llama3.2:3b", agente_cfg["modelo_rapido"]]
            modelo_profundo = next(
                (m for m in candidatos if verificar_modelo_disponivel(m)),
                agente_cfg["modelo_rapido"],
            )
            console.print(
                f"[yellow]⚠️  Modelo profundo '{agente_cfg['modelo_profundo']}' não instalado. "
                f"Usando '{modelo_profundo}' como fallback.[/yellow]"
            )

        resultado = chamar_llm(
            modelo=modelo_profundo,
            system_prompt=self._system_prompt_com_rag(agente_cfg["system_prompt"], bool(contexto_rag)),
            mensagens=mensagens,
            stream=True,
            max_tokens=2048,
            temperatura=0.7,
            num_ctx=NUM_CTX_NIVEL[3],
            keep_alive=KEEP_ALIVE_PRINCIPAL,
        )

        self._salvar(pergunta, resultado["resposta"], nome_agente, nivel=3, inicio=inicio)
        self._salvar_metrica(
            nome_agente, 3, inicio,
            tokens_in=resultado["tokens_entrada"],
            tokens_out=resultado["tokens_saida"],
            fonte="llm_profundo",
        )

        # Resumo automático
        if self.memoria.total_mensagens() % 10 == 0:
            self._gerar_resumo(agente_cfg["modelo_rapido"])

        resposta_final = agente_obj.pos_execucao(pergunta, resultado["resposta"])
        return resposta_final

    def _system_prompt_com_rag(self, base_prompt: str, tem_rag: bool) -> str:
        """Aplica instruções de grounding quando houver contexto recuperado."""
        if not tem_rag:
            return base_prompt
        complemento = (
            "\n\n## REGRAS DE GROUNDING (OBRIGATÓRIAS):\n"
            "1. Responda EXCLUSIVAMENTE com base no contexto fornecido abaixo.\n"
            "2. Se o contexto não contém a informação necessária, diga explicitamente: "
            "'Não encontrei essa informação nas fontes consultadas.'\n"
            "3. NUNCA invente dados, estatísticas, datas ou fatos não presentes no contexto.\n"
            "4. Cite a fonte quando possível (URL ou título do documento).\n"
            "5. Se houver conflito entre fontes, mencione ambas perspectivas.\n"
            "6. Prefira responder com bullet points para clareza."
        )
        return f"{base_prompt}{complemento}"

    def _construir_contexto_rag(self, docs: list[dict]) -> str:
        """Compacta o contexto RAG para reduzir ruído sem perder cobertura."""
        blocos: list[str] = []
        total_chars = 0
        for idx, doc in enumerate(docs, 1):
            score = doc.get("score_hibrido", doc.get("similaridade", 0.0))
            conteudo = doc.get("conteudo", "").strip()
            if not conteudo:
                continue

            # Mantém o contexto dentro de um limite para proteger qualidade no modelo menor.
            trecho = conteudo[:900]
            bloco = f"[Doc {idx} | Score {score:.0%}]\n{trecho}"

            if total_chars + len(bloco) > RAG_MAX_CHARS:
                break

            blocos.append(bloco)
            total_chars += len(bloco)

        return "\n\n".join(blocos)

    def _executar_ferramenta_por_intencao(self, intencao: IntencaoAnalisada, pergunta: str) -> str | None:
        """
        Executa ferramenta baseado na análise de intenção da LLM.
        A LLM já entendeu semanticamente o que o usuário quer — não precisa de keywords.
        """
        if not intencao.ferramenta:
            return None

        params = intencao.parametros

        if intencao.ferramenta == "data_hora":
            local = params.get("local", "")
            if local:
                # Tenta resolver timezone pelo nome do local
                timezone = _extrair_local_hora(f"hora em {local}")
                return obter_data_hora(timezone)
            return obter_data_hora()

        if intencao.ferramenta == "calculo":
            expressao = params.get("expressao", "")
            if expressao:
                resultado = calcular(expressao)
                if resultado:
                    return resultado
            # Fallback: tenta extrair expressão da pergunta original
            return None

        if intencao.ferramenta == "saudacao":
            return verificar_ferramenta_saudacao(pergunta) or (
                "Olá! Estou pronto para ajudar. "
                "Você pode pedir código, análise, pesquisa ou execução de comandos."
            )

        if intencao.ferramenta == "arquivo":
            acao = params.get("acao", "")
            caminho = params.get("caminho", "")
            if acao == "listar":
                return listar_pasta(caminho or ".")
            elif acao == "ler" and caminho:
                return ler_arquivo(caminho)
            elif acao == "criar" and caminho:
                conteudo = params.get("conteudo", "")
                return criar_arquivo(caminho, conteudo)
            return None

        if intencao.ferramenta == "comando":
            comando = params.get("comando", "")
            if comando:
                return executar_comando_local(comando)
            return None

        return None

    @staticmethod
    def _deve_promover_para_profundo(pergunta: str, resposta: str) -> bool:
        """Promove para nível profundo quando a saída rápida é fraca para a pergunta."""
        if not resposta:
            return True

        resposta_lower = resposta.lower().strip()
        pergunta_tokens = len(pergunta.split())

        sinais_incerteza = [
            "não sei",
            "não tenho informação",
            "não encontrei",
            "não posso afirmar",
            "não posso fornecer",
            "não consigo fornecer",
            "não tenho acesso",
            "não possuo acesso",
            "recomendo consultar",
            "talvez",
            "depende",
        ]
        if any(s in resposta_lower for s in sinais_incerteza):
            return True

        if pergunta_tokens >= 14 and len(resposta_lower) < 120:
            return True

        return False

    def _montar_contexto(
        self, n_msgs: int, contexto_busca: str, pergunta: str, contexto_rag: str = ""
    ) -> list[dict]:
        """Monta lista de mensagens para o LLM."""
        mensagens = []

        resumo = self.memoria.ultimo_resumo()
        if resumo:
            mensagens.append({
                "role": "system",
                "content": f"Contexto anterior: {resumo}"
            })

        if contexto_rag:
            mensagens.append({
                "role": "system",
                "content": f"Conhecimento relevante da base:\n{contexto_rag}"
            })

        # Resultados web como system message — garante que modelos pequenos não ignorem
        if contexto_busca:
            mensagens.append({
                "role": "system",
                "content": (
                    "DADOS ATUAIS OBTIDOS DA WEB — use exclusivamente estes dados para responder. "
                    "NUNCA diga que não tem acesso a informações em tempo real quando estes dados estiverem presentes.\n\n"
                    f"{contexto_busca}"
                ),
            })

        historico = self.memoria.ultimas_mensagens(n_msgs)
        for msg in historico:
            mensagens.append({"role": msg["role"], "content": msg["content"]})

        # Para modelos pequenos: inclui dados também na mensagem user para garantir grounding
        if contexto_busca:
            conteudo = (
                f"{pergunta}\n\n"
                f"[Dados obtidos agora da web — responda com base exclusivamente nestes dados]:\n"
                f"{contexto_busca[:1500]}"
            )
        else:
            conteudo = pergunta

        mensagens.append({"role": "user", "content": conteudo})

        return mensagens

    def _salvar(self, pergunta: str, resposta: str, agente: str, nivel: int, inicio: float):
        """Salva em todas as camadas de memória (ignora respostas de erro)."""
        # P0: Não cachear respostas de erro do LLM
        if resposta.startswith("Erro ao chamar modelo") or resposta.startswith("Erro"):
            self.memoria.salvar_mensagem("user", pergunta)
            self._salvar_metrica(agente, nivel, inicio, fonte="erro")
            return

        self.memoria.salvar_mensagem("user", pergunta)
        self.memoria.salvar_mensagem("assistant", resposta, agente, nivel)
        cache_key = f"{agente}:{pergunta}"
        self.cache.salvar(cache_key, resposta, agente)
        self.semantica.adicionar(pergunta, resposta, agente)

    def _salvar_metrica(self, agente: str, nivel: int, inicio: float,
                        tokens_in: int = 0, tokens_out: int = 0, fonte: str = ""):
        """Salva métrica de performance."""
        tempo_ms = int((time.time() - inicio) * 1000)
        self.memoria.salvar_metrica(agente, nivel, tempo_ms, tokens_in, tokens_out, fonte)

    def _gerar_resumo(self, modelo: str):
        """Gera resumo automático."""
        mensagens = self.memoria.ultimas_mensagens(10)
        if mensagens:
            console.print("[dim]📝 Resumo automático...[/dim]")
            resumo = resumir_conversa(modelo, mensagens)
            if resumo:
                self.memoria.salvar_resumo(resumo)

    def forcar_nivel(self, nivel: int):
        """Força nível para a próxima pergunta."""
        if nivel in (1, 2, 3):
            self.nivel_forcado = nivel

    def estatisticas(self) -> dict:
        """Retorna estatísticas completas."""
        return {
            "cache": self.cache.estatisticas(),
            "chromadb": self.semantica.estatisticas(),
            "metricas": self.memoria.metricas_resumo(),
            "mensagens_total": self.memoria.total_mensagens(),
        }

    def ingerir_conhecimento(self, texto: str, fonte: str = ""):
        """Adiciona conhecimento à base vetorial."""
        self.semantica.adicionar_conhecimento(texto, fonte)
        console.print(f"[green]📚 Conhecimento adicionado ({len(texto)} chars)[/green]")

    def fechar(self):
        self.memoria.fechar()
