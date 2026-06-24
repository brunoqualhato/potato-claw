"""
Motor de execução dos agentes com 3 níveis de performance.

Pipeline por nível:
  Nível 1: Ferramentas → Cache → ChromaDB → RETORNA (sem LLM!)
  Nível 2: ...nivel1... → Modelo Rápido (1.7B) + contexto mínimo
  Nível 3: ...nivel1... → ChromaDB RAG → Modelo Completo (4B) + contexto rico

Otimizações para modelos pequenos:
  - Self-correction loop antes de promover nível
  - Few-shot dinâmico salva classificações bem-sucedidas
  - Prompts compactos por nível
  - Detecção de degeneração no streaming
  - Feedback do usuário salvo como preferência
"""

import logging
import time

from rich.console import Console
from rich.panel import Panel

from src.agentes.base import Agente, ConfigAgente, criar_agente
from src.agentes.coordenador import rotear_por_palavras_chave
from src.core.analisador import IntencaoAnalisada, analisar_intencao, salvar_intencao_classificada
from src.core.classificador import classificar_complexidade, explicar_nivel
from src.core.config import (
    AGENTES,
    AUTO_APROVAR_ACOES_LOCAIS,
    CHROMADB_NIVEL1_THRESHOLD,
    CONTEXTO_MAX_MSGS,
    KEEP_ALIVE_PRINCIPAL,
    NUM_CTX_NIVEL,
    PERFIL_ATIVO,
    RAG_MAX_CHARS,
    RAG_MAX_DOCS,
    RAM_GB,
)
from src.core.llm import chamar_llm, resumir_conversa, verificar_modelo_disponivel
from src.ferramentas.resolver import (
    _extrair_local_hora,
    calcular,
    criar_arquivo,
    descrever_acao_local_mutavel,
    executar_comando_local,
    executar_ferramentas,
    ler_arquivo,
    listar_pasta,
    obter_data_hora,
    remover_confirmacao,
    verificar_ferramenta_saudacao,
)
from src.ferramentas.web_async import paralelo_sync
from src.ferramentas.web_rag import pesquisar_web_profunda, pesquisar_web_rapida
from src.memoria.cache import Cache
from src.memoria.semantica import MemoriaSemantica
from src.memoria.sqlite import Memoria

logger = logging.getLogger(__name__)
console = Console()


class SistemaAgentes:
    """Gerencia a execução dos agentes com 3 níveis de performance."""

    def __init__(self, memoria=None, cache=None, semantica=None):
        self.memoria = memoria or Memoria()
        self.cache = cache or Cache()
        self.semantica = semantica or MemoriaSemantica()
        self.nivel_forcado: int | None = None
        self.ultimo_agente = "generalista"
        self.ultimo_nivel = 0
        self.ultima_fonte = ""
        self._acao_local_confirmada = False
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

    # ══════════════════════════════════════════════════════════════
    # PIPELINE PRINCIPAL
    # ══════════════════════════════════════════════════════════════

    def executar(self, nome_agente: str, pergunta: str) -> str:
        """
        Pipeline principal com análise de intenção via LLM.

        Fluxo:
          1. LLM analisa intenção (agente, web, ferramenta, params) — ~100ms
          2. Se ferramenta local resolve → retorna sem LLM (nível 1)
          3. Se precisa web → fetch + extract primeiro
          4. LLM final responde grounded nos dados coletados
        """
        inicio = time.time()

        # Ferramentas determinísticas podem rodar antes da análise de intenção.
        # O cache só é consultado depois, quando sabemos se a pergunta exige
        # dados atuais (`precisa_web`).
        pergunta, self._acao_local_confirmada = remover_confirmacao(pergunta)
        resultado_preflight = self._preflight_deterministico(
            nome_agente, pergunta, inicio
        )
        if resultado_preflight:
            return resultado_preflight

        # Só usa a LLM coordenadora quando os caminhos baratos não resolveram.
        intencao, nome_agente, agente_cfg, agente_obj = self._analisar_e_rotear(
            nome_agente, pergunta
        )

        # Classificar complexidade e aplicar hooks do agente
        nivel = self._classificar_nivel(pergunta, agente_cfg)
        pergunta, nivel = agente_obj.pre_execucao(pergunta, nivel)
        console.print(f"[dim]{explicar_nivel(nivel)}[/dim]")

        # Pipeline nível 1: ferramentas, cache, ChromaDB
        resultado_n1, docs_similares = self._pipeline_nivel1(
            intencao, pergunta, nome_agente, nivel, inicio
        )
        if resultado_n1:
            return resultado_n1

        # Promover se nível 1 não resolveu
        if nivel == 1:
            nivel = 2
            console.print("[dim]↑ Promovido para Nível 2[/dim]")

        # Pipeline nível 2: modelo rápido
        if nivel == 2:
            resultado_n2 = self._pipeline_nivel2(
                intencao, pergunta, nome_agente, agente_cfg, agente_obj, inicio
            )
            if resultado_n2:
                return resultado_n2
            # Promoção automática para nível 3
            nivel = 3

        # Pipeline nível 3: modelo profundo + RAG
        return self._pipeline_nivel3(
            intencao, pergunta, nome_agente, agente_cfg, agente_obj,
            docs_similares, inicio
        )

    def _preflight_deterministico(
        self, nome_agente: str, pergunta: str, inicio: float
    ) -> str | None:
        """Resolve ações óbvias sem carregar qualquer modelo."""
        agente_heuristico = (
            nome_agente
            if nome_agente != "generalista"
            else rotear_por_palavras_chave(pergunta) or "generalista"
        )

        acao_mutavel = descrever_acao_local_mutavel(pergunta)
        if (
            acao_mutavel
            and not self._acao_local_confirmada
            and not AUTO_APROVAR_ACOES_LOCAIS
        ):
            resposta = (
                f"Confirmação necessária para {acao_mutavel}. "
                f"Repita a solicitação começando com `confirmar:`."
            )
            self._registrar_execucao(agente_heuristico, 1, "confirmacao")
            self._salvar_confirmacao(pergunta, resposta, agente_heuristico, inicio)
            return resposta

        resultado_ferramenta = executar_ferramentas(pergunta)
        if resultado_ferramenta:
            console.print(
                Panel(resultado_ferramenta, title="⚡ Nível 1 • Ferramenta", border_style="cyan")
            )
            self._registrar_execucao(agente_heuristico, 1, "ferramenta")
            self._salvar(
                pergunta, resultado_ferramenta, agente_heuristico, nivel=1, inicio=inicio
            )
            return resultado_ferramenta

        return None

    # ══════════════════════════════════════════════════════════════
    # ANÁLISE E ROTEAMENTO
    # ══════════════════════════════════════════════════════════════

    def _analisar_e_rotear(
        self, nome_agente: str, pergunta: str
    ) -> tuple[IntencaoAnalisada, str, dict, Agente]:
        """Analisa intenção e redireciona agente se necessário."""
        intencao = analisar_intencao(pergunta)
        console.print(
            f"[dim]🧠 Intenção: agente={intencao.agente}, web={intencao.precisa_web}, "
            f"ferramenta={intencao.ferramenta}[/dim]"
        )

        # Redireciona se analisador sugeriu agente diferente
        if nome_agente == "generalista" and intencao.agente != "generalista":
            nome_agente = intencao.agente
            console.print(f"[dim]🎯 {nome_agente}[/dim]")

        if nome_agente not in AGENTES:
            nome_agente = "generalista"
        agente_cfg = AGENTES[nome_agente]
        agente_obj = self._obter_agente(nome_agente)
        self.ultimo_agente = nome_agente
        return intencao, nome_agente, agente_cfg, agente_obj

    def _classificar_nivel(self, pergunta: str, agente_cfg: dict) -> int:
        """Determina o nível de execução baseado em forçamento ou heurística."""
        if self.nivel_forcado:
            nivel = self.nivel_forcado
            self.nivel_forcado = None
            return nivel

        nivel = classificar_complexidade(pergunta)
        if nivel == 2 and agente_cfg.get("nivel_preferido", 2) == 3:
            if len(pergunta.split()) > 12:
                nivel = 3
        return nivel

    # ══════════════════════════════════════════════════════════════
    # PIPELINE NÍVEL 1: FERRAMENTAS LOCAIS (sem LLM)
    # ══════════════════════════════════════════════════════════════

    def _pipeline_nivel1(
        self,
        intencao: IntencaoAnalisada,
        pergunta: str,
        nome_agente: str,
        nivel: int,
        inicio: float,
    ) -> tuple[str | None, list[dict]]:
        """
        Tenta resolver sem LLM: ferramenta, cache, ChromaDB.
        Retorna (resposta_ou_None, docs_similares_para_reuso).
        """
        # 1a. Ferramenta identificada pelo analisador
        resultado_ferramenta = self._executar_ferramenta_por_intencao(intencao, pergunta)
        if resultado_ferramenta:
            console.print(
                Panel(resultado_ferramenta, title="⚡ Nível 1 • Ferramenta", border_style="cyan")
            )
            if resultado_ferramenta.startswith("Confirmação necessária"):
                self._salvar_confirmacao(
                    pergunta, resultado_ferramenta, nome_agente, inicio
                )
                self._registrar_execucao(nome_agente, 1, "confirmacao")
            else:
                self._salvar(
                    pergunta, resultado_ferramenta, nome_agente, nivel=1, inicio=inicio
                )
                self._registrar_execucao(nome_agente, 1, "ferramenta")
            return resultado_ferramenta, []

        # 1b. Cache exato — ignorado se precisa de dados frescos
        cache_key = f"{nome_agente}:{pergunta}"
        resposta_cache = None if intencao.precisa_web else self.cache.buscar(cache_key)
        if resposta_cache:
            console.print("[dim]📋 Nível 1 • Cache[/dim]")
            console.print(resposta_cache, style="green")
            self._salvar_metrica(nome_agente, 1, inicio, fonte="cache")
            self._registrar_execucao(nome_agente, 1, "cache")
            return resposta_cache, []

        # 1c. ChromaDB — busca semântica
        docs_similares = self.semantica.buscar_similar(pergunta)
        if docs_similares and nivel == 1 and not intencao.precisa_web:
            melhor = docs_similares[0]
            score = melhor.get("score_hibrido", melhor["similaridade"])
            if score >= CHROMADB_NIVEL1_THRESHOLD:
                resposta = melhor["conteudo"].split("Resposta: ", 1)[-1]
                console.print(f"[dim]🧲 Nível 1 • ChromaDB ({score:.0%})[/dim]")
                console.print(resposta, style="green")
                self._salvar_metrica(nome_agente, 1, inicio, fonte="chromadb")
                self._registrar_execucao(nome_agente, 1, "chromadb")
                return resposta, docs_similares

        return None, docs_similares

    # ══════════════════════════════════════════════════════════════
    # PIPELINE NÍVEL 2: RÁPIDO (modelo leve)
    # ══════════════════════════════════════════════════════════════

    def _pipeline_nivel2(
        self,
        intencao: IntencaoAnalisada,
        pergunta: str,
        nome_agente: str,
        agente_cfg: dict,
        agente_obj: Agente,
        inicio: float,
    ) -> str | None:
        """
        Executa com modelo rápido. Retorna resposta ou None se deve promover.
        Inclui self-correction loop antes de promover para nível 3.
        """
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

        self._salvar_metrica(
            nome_agente, 2, inicio,
            tokens_in=resultado["tokens_entrada"],
            tokens_out=resultado["tokens_saida"],
            fonte="llm_rapido",
        )

        if self._deve_promover_para_profundo(pergunta, resultado["resposta"]):
            # Self-correction: tenta corrigir no mesmo modelo antes de promover
            corrigida = self._autocorrecao_rapida(
                pergunta, resultado["resposta"], agente_cfg["modelo_rapido"]
            )
            if corrigida and not self._deve_promover_para_profundo(pergunta, corrigida):
                console.print("[dim]🔄 Autocorreção bem-sucedida[/dim]")
                self._salvar(pergunta, corrigida, nome_agente, nivel=2, inicio=inicio)
                resposta_final = agente_obj.pos_execucao(pergunta, corrigida)
                # Salva classificação como bem-sucedida
                salvar_intencao_classificada(pergunta, intencao)
                self._registrar_execucao(nome_agente, 2, "llm_rapido_autocorrecao")
                return resposta_final

            console.print("[dim]↑ Ajuste de precisão: promovido para Nível 3[/dim]")
            return None  # Sinaliza promoção

        self._salvar(pergunta, resultado["resposta"], nome_agente, nivel=2, inicio=inicio)
        resposta_final = agente_obj.pos_execucao(pergunta, resultado["resposta"])
        # Salva classificação como bem-sucedida
        salvar_intencao_classificada(pergunta, intencao)
        self._registrar_execucao(nome_agente, 2, "llm_rapido")
        return resposta_final

    def _autocorrecao_rapida(self, pergunta: str, resposta_fraca: str, modelo: str) -> str | None:
        """
        Self-correction loop: tenta corrigir resposta fraca no mesmo modelo.
        Custo: ~80-150ms com modelo 1.2B. Evita promoção desnecessária ao nível 3.
        """
        try:
            import ollama
            response = ollama.chat(
                model=modelo,
                messages=[
                    {"role": "system", "content": (
                        "Sua resposta anterior foi vaga ou incompleta."
                        " Responda novamente de forma precisa e direta."
                    )},
                    {"role": "user", "content": pergunta},
                    {"role": "assistant", "content": resposta_fraca},
                    {"role": "user", "content": "Responda melhor, com dados concretos. Seja específico."},
                ],
                options={"temperature": 0.3, "num_predict": 512},
            )
            corrigida = response["message"]["content"].strip()
            if corrigida and len(corrigida) > len(resposta_fraca) * 0.5:
                return corrigida
        except Exception as e:
            logger.debug("Autocorreção falhou: %s", e)
        return None

    # ══════════════════════════════════════════════════════════════
    # PIPELINE NÍVEL 3: PROFUNDO (modelo grande + RAG)
    # ══════════════════════════════════════════════════════════════

    def _pipeline_nivel3(
        self,
        intencao: IntencaoAnalisada,
        pergunta: str,
        nome_agente: str,
        agente_cfg: dict,
        agente_obj: Agente,
        docs_similares: list[dict],
        inicio: float,
    ) -> str:
        """Executa com modelo profundo, RAG e web search completo."""
        contexto_busca, contexto_rag = self._obter_contextos_nivel3(
            intencao, pergunta, docs_similares
        )

        mensagens = self._montar_contexto(
            n_msgs=min(5, CONTEXTO_MAX_MSGS),
            contexto_busca=contexto_busca,
            pergunta=pergunta,
            contexto_rag=contexto_rag,
        )

        modelo_profundo = self._resolver_modelo_profundo(agente_cfg)

        resultado = chamar_llm(
            modelo=modelo_profundo,
            system_prompt=self._system_prompt_com_rag(
                agente_cfg["system_prompt"], bool(contexto_rag)
            ),
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

        resposta_final = agente_obj.pos_execucao(pergunta, resultado["resposta"])

        # Resumo automático a cada 10 mensagens
        if self.memoria.total_mensagens() % 10 == 0:
            self._gerar_resumo(agente_cfg["modelo_rapido"])

        # Salva classificação como bem-sucedida para few-shot futuro
        salvar_intencao_classificada(pergunta, intencao)
        self._registrar_execucao(nome_agente, 3, "llm_profundo")

        return resposta_final

    def _obter_contextos_nivel3(
        self,
        intencao: IntencaoAnalisada,
        pergunta: str,
        docs_similares: list[dict],
    ) -> tuple[str, str]:
        """Obtém contexto web e RAG, paralelizando quando possível."""
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

        return contexto_busca, contexto_rag

    def _resolver_modelo_profundo(self, agente_cfg: dict) -> str:
        """Resolve qual modelo profundo usar, com fallback automático."""
        modelo_profundo = agente_cfg["modelo_profundo"]
        if verificar_modelo_disponivel(modelo_profundo):
            return modelo_profundo

        candidatos = ["qwen3:4b", "qwen2.5:3b", "llama3.2:3b", agente_cfg["modelo_rapido"]]
        modelo_profundo = next(
            (m for m in candidatos if verificar_modelo_disponivel(m)),
            agente_cfg["modelo_rapido"],
        )
        console.print(
            f"[yellow]⚠️  Modelo profundo '{agente_cfg['modelo_profundo']}' não instalado. "
            f"Usando '{modelo_profundo}' como fallback.[/yellow]"
        )
        return modelo_profundo

    # ══════════════════════════════════════════════════════════════
    # FERRAMENTAS
    # ══════════════════════════════════════════════════════════════

    def _executar_ferramenta_por_intencao(
        self, intencao: IntencaoAnalisada, pergunta: str
    ) -> str | None:
        """
        Executa ferramenta baseado na análise de intenção da LLM.
        A LLM já entendeu semanticamente o que o usuário quer.
        """
        if not intencao.ferramenta:
            return None

        params = intencao.parametros

        if (
            intencao.ferramenta in {"arquivo", "comando"}
            and not self._acao_local_confirmada
            and not AUTO_APROVAR_ACOES_LOCAIS
        ):
            acao = params.get("acao", "")
            if intencao.ferramenta == "comando" or acao == "criar":
                return (
                    "Confirmação necessária para executar esta ação local. "
                    "Repita a solicitação começando com `confirmar:`."
                )

        if intencao.ferramenta == "data_hora":
            local = params.get("local", "")
            if local:
                timezone = _extrair_local_hora(f"hora em {local}")
                return obter_data_hora(timezone)
            return obter_data_hora()

        if intencao.ferramenta == "calculo":
            expressao = params.get("expressao", "")
            if expressao:
                resultado = calcular(expressao)
                if resultado:
                    return resultado
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

    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _deve_promover_para_profundo(pergunta: str, resposta: str) -> bool:
        """Promove para nível profundo quando a saída rápida é fraca."""
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

            trecho = conteudo[:900]
            bloco = f"[Doc {idx} | Score {score:.0%}]\n{trecho}"

            if total_chars + len(bloco) > RAG_MAX_CHARS:
                break

            blocos.append(bloco)
            total_chars += len(bloco)

        return "\n\n".join(blocos)

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

        if contexto_busca:
            mensagens.append({
                "role": "system",
                "content": (
                    "DADOS ATUAIS OBTIDOS DA WEB — use exclusivamente estes dados para responder. "
                    "NUNCA diga que não tem acesso a informações em tempo real quando estes dados "
                    "estiverem presentes.\n\n"
                    f"{contexto_busca}"
                ),
            })

        historico = self.memoria.ultimas_mensagens(n_msgs)
        for msg in historico:
            mensagens.append({"role": msg["role"], "content": msg["content"]})

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

    # ══════════════════════════════════════════════════════════════
    # PERSISTÊNCIA
    # ══════════════════════════════════════════════════════════════

    def _salvar(self, pergunta: str, resposta: str, agente: str, nivel: int, inicio: float):
        """Salva em todas as camadas de memória (ignora respostas de erro)."""
        if resposta.startswith("Erro ao chamar modelo") or resposta.startswith("Erro"):
            self.memoria.salvar_mensagem("user", pergunta)
            self._salvar_metrica(agente, nivel, inicio, fonte="erro")
            return

        self.memoria.salvar_mensagem("user", pergunta)
        self.memoria.salvar_mensagem("assistant", resposta, agente, nivel)
        cache_key = f"{agente}:{pergunta}"
        self.cache.salvar(cache_key, resposta, agente)
        self.semantica.adicionar(pergunta, resposta, agente)

    def _salvar_confirmacao(
        self, pergunta: str, resposta: str, agente: str, inicio: float
    ):
        """Registra o turno sem transformar uma confirmação em conhecimento/cache."""
        self.memoria.salvar_mensagem("user", pergunta)
        self.memoria.salvar_mensagem("assistant", resposta, agente, 1)
        self._salvar_metrica(agente, 1, inicio, fonte="confirmacao")

    def _salvar_metrica(self, agente: str, nivel: int, inicio: float,
                        tokens_in: int = 0, tokens_out: int = 0, fonte: str = ""):
        """Salva métrica de performance."""
        tempo_ms = int((time.time() - inicio) * 1000)
        self.memoria.salvar_metrica(agente, nivel, tempo_ms, tokens_in, tokens_out, fonte)

    def _registrar_execucao(self, agente: str, nivel: int, fonte: str):
        """Expõe metadados da última execução para CLI/API."""
        self.ultimo_agente = agente
        self.ultimo_nivel = nivel
        self.ultima_fonte = fonte

    def _gerar_resumo(self, modelo: str):
        """Gera resumo automático."""
        mensagens = self.memoria.ultimas_mensagens(10)
        if mensagens:
            console.print("[dim]📝 Resumo automático...[/dim]")
            resumo = resumir_conversa(modelo, mensagens)
            if resumo:
                self.memoria.salvar_resumo(resumo)

    # ══════════════════════════════════════════════════════════════
    # INTERFACE PÚBLICA
    # ══════════════════════════════════════════════════════════════

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
            "metricas_por_fonte": self.memoria.metricas_por_fonte(),
            "mensagens_total": self.memoria.total_mensagens(),
            "hardware": {"ram_gb": RAM_GB, "perfil": PERFIL_ATIVO},
        }

    def ingerir_conhecimento(self, texto: str, fonte: str = ""):
        """Adiciona conhecimento à base vetorial."""
        self.semantica.adicionar_conhecimento(texto, fonte)
        console.print(f"[green]📚 Conhecimento adicionado ({len(texto)} chars)[/green]")

    def salvar_feedback(self, pergunta: str, feedback: str):
        """
        Salva feedback/correção do usuário como preferência permanente.
        Permite que o sistema aprenda padrões do usuário real.
        """
        doc = f"PREFERÊNCIA: Quando peço '{pergunta}', quero '{feedback}'"
        self.semantica.adicionar_conhecimento(doc, fonte="feedback_usuario", tipo="preferencia")
        console.print("[dim]📝 Preferência salva[/dim]")

    def fechar(self):
        self.memoria.fechar()
