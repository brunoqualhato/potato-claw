"""
Analisador de Intenção via LLM.

Em vez de comparar palavras-chave, a LLM coordenadora analisa
semanticamente a pergunta e decide:
  - Qual agente usar
  - Se precisa buscar na web
  - Se pode resolver com ferramenta local (cálculo, hora, arquivo)
  - Parâmetros extraídos (local, expressão, etc)

Uma única chamada de ~100ms substitui toda a lógica de keyword matching.

Otimizações para modelos pequenos (1.2B):
  - format="json" garante output estruturado sem desperdício de tokens
  - Prompt compacto (~300 tokens) libera espaço para contexto
  - Few-shot dinâmico via ChromaDB quando disponível
"""

from __future__ import annotations

import json
import logging
import re

from src.core.config import (
    COORDENADOR_MODELO,
    KEEP_ALIVE_EFEMERO,
    KEEP_ALIVE_PRINCIPAL,
    MODELOS,
    NUM_CTX_AUXILIAR,
)
from src.core.llm import chamar_llm
from src.core.utils import normalizar

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# PROMPT DO ANALISADOR (compacto para modelos 1.2B)
# ══════════════════════════════════════════════════════════════

_PROMPT_ANALISAR = """Classifique a intenção do usuário em JSON com estes campos:
- "agente": generalista|programador|pesquisador|analista
- "precisa_web": true se precisa de dados atuais (clima, cotação, versões, notícias, docs)
- "ferramenta": null|calculo|data_hora|saudacao|arquivo|comando
- "parametros": dados extraídos ({} se vazio)

Regras:
- generalista: dúvidas cotidianas, culinária, receitas e conhecimento estável
- programador: SOMENTE quando a pergunta mencionar software, código ou tecnologia
- calculo: parametros.expressao com a expressão matemática
- data_hora: parametros.local se pede hora de um lugar
- arquivo: parametros.acao (ler|criar|listar) + parametros.caminho
- comando: parametros.comando
- saudacao: para oi/hello/bom dia
- precisa_web=true SOMENTE com busca explícita ou informação temporal/atual
- precisa_web=false se é conhecimento estável, receita ou ferramenta local resolve"""


# ══════════════════════════════════════════════════════════════
# TIPOS
# ══════════════════════════════════════════════════════════════

class IntencaoAnalisada:
    """Resultado da análise de intenção."""

    __slots__ = ("agente", "precisa_web", "ferramenta", "parametros", "_raw")

    def __init__(
        self,
        agente: str = "generalista",
        precisa_web: bool = False,
        ferramenta: str | None = None,
        parametros: dict | None = None,
        raw: str = "",
    ):
        self.agente = agente
        self.precisa_web = precisa_web
        self.ferramenta = ferramenta
        self.parametros = parametros or {}
        self._raw = raw

    def __repr__(self) -> str:
        return (
            f"Intencao(agente={self.agente}, web={self.precisa_web}, "
            f"ferramenta={self.ferramenta}, params={self.parametros})"
        )


# ══════════════════════════════════════════════════════════════
# FEW-SHOT DINÂMICO
# ══════════════════════════════════════════════════════════════

def _obter_exemplos_dinamicos(pergunta: str) -> str:
    """
    Busca exemplos de classificação anteriores similares no ChromaDB.
    Retorna string com 2-3 exemplos ou string vazia.
    """
    try:
        from src.memoria.semantica import MemoriaSemantica
        sem = MemoriaSemantica()
        docs = sem.buscar_similar(
            f"intencao:{pergunta}",
            top_k=3,
            tipos=("intencao_classificada",),
        )
        exemplos = []
        for doc in docs:
            meta = doc.get("metadata", {})
            if meta.get("tipo") == "intencao_classificada":
                exemplos.append(doc["conteudo"])
        if exemplos:
            return "\nExemplos similares:\n" + "\n".join(exemplos[:3])
    except Exception:
        pass
    return ""


def salvar_intencao_classificada(pergunta: str, intencao: "IntencaoAnalisada"):
    """
    Salva uma classificação bem-sucedida para uso futuro como few-shot.
    Chamado pelo executor após confirmar que a classificação funcionou.
    """
    try:
        from src.memoria.semantica import MemoriaSemantica
        sem = MemoriaSemantica()
        doc = (
            f'Pergunta: "{pergunta}"\n'
            f'{{"agente":"{intencao.agente}","precisa_web":{str(intencao.precisa_web).lower()},'
            f'"ferramenta":{json.dumps(intencao.ferramenta)},"parametros":{json.dumps(intencao.parametros)}}}'
        )
        sem.adicionar_conhecimento(
            texto=doc,
            fonte="analisador",
            tipo="intencao_classificada",
        )
    except Exception as e:
        logger.debug("Erro ao salvar intenção classificada: %s", e)


# ══════════════════════════════════════════════════════════════
# ANALISADOR
# ══════════════════════════════════════════════════════════════

_AGENTES_VALIDOS = {"generalista", "programador", "pesquisador", "analista"}
_FERRAMENTAS_VALIDAS = {"calculo", "data_hora", "saudacao", "arquivo", "comando", None}

_SINAIS_TECNICOS = {
    "api", "app", "aplicacao", "algoritmo", "backend", "banco", "bug",
    "autenticacao", "classe", "codigo", "crud", "css", "deploy", "docker",
    "endpoint", "erro", "framework", "frontend", "funcao", "git", "html",
    "implementar", "javascript", "jwt", "programa", "python", "refatorar",
    "script", "servidor", "site", "software", "sql", "sistema", "teste",
    "typescript",
}
_SINAIS_ATUAIS = {
    "agora", "atual", "cotacao", "clima", "hoje", "latest", "noticia",
    "preco", "previsao", "recente", "release", "temperatura", "ultima",
    "ultimo", "versao",
}
_SINAIS_BUSCA_EXPLICITA = {"busque", "buscar", "pesquise", "pesquisar", "procure", "procurar"}
_SINAIS_CONSULTA_ESTAVEL = {
    "como fazer", "como preparar", "defina", "explique", "o que e", "receita",
}
_SINAIS_CULINARIOS = {
    "assar", "bolo", "cozinhar", "forno", "ingrediente", "massa", "molho",
    "pao", "queijo", "receita",
}


def _tokens_normalizados(texto: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_-]{2,}", normalizar(texto)))


def _tem_sinal_atual(texto: str) -> bool:
    texto_norm = normalizar(texto)
    tokens = _tokens_normalizados(texto)
    return (
        any(sinal in texto_norm for sinal in _SINAIS_ATUAIS)
        or bool(tokens & _SINAIS_BUSCA_EXPLICITA)
        or bool(re.search(r"\b20\d{2}\b", texto_norm))
    )


def _corrigir_inconsistencias(
    pergunta: str, intencao: IntencaoAnalisada
) -> IntencaoAnalisada:
    """Aplica guardrails baratos sobre classificações improváveis de modelos pequenos."""
    texto_norm = normalizar(pergunta)
    tokens = _tokens_normalizados(pergunta)

    # "Como fazer" não implica programação. Só mantém o especialista técnico
    # quando a própria pergunta contém algum sinal verificável desse domínio.
    if intencao.agente == "programador" and not (tokens & _SINAIS_TECNICOS):
        intencao.agente = "generalista"

    eh_culinaria = bool(tokens & _SINAIS_CULINARIOS)
    consulta_estavel = any(sinal in texto_norm for sinal in _SINAIS_CONSULTA_ESTAVEL)
    if eh_culinaria:
        # O modelo ultraleve não é confiável para ingredientes/quantidades.
        # Grounding custa uma busca, mas evita receitas inventadas.
        intencao.precisa_web = True
        intencao.parametros["consulta_web"] = (
            f"receita tradicional {pergunta} ingredientes modo de preparo"
        )
        intencao.parametros["resposta_web_direta"] = True
    elif intencao.precisa_web and consulta_estavel and not _tem_sinal_atual(pergunta):
        intencao.precisa_web = False

    return intencao


def analisar_intencao(pergunta: str, modelo: str = COORDENADOR_MODELO) -> IntencaoAnalisada:
    """
    Usa a LLM coordenadora para analisar semanticamente a pergunta.
    Retorna IntencaoAnalisada com agente, web, ferramenta e parâmetros.

    Otimizações:
      - format="json" → garante output JSON válido direto do Ollama
      - Few-shot dinâmico → exemplos relevantes do ChromaDB
      - Prompt compacto → ~300 tokens liberando espaço para contexto
    """
    try:
        # Few-shot dinâmico (se disponível)
        exemplos = _obter_exemplos_dinamicos(pergunta)
        prompt_completo = _PROMPT_ANALISAR + exemplos

        keep_alive = (
            KEEP_ALIVE_PRINCIPAL
            if modelo == MODELOS["rapido"]
            else KEEP_ALIVE_EFEMERO
        )
        resultado = chamar_llm(
            modelo=modelo,
            system_prompt=prompt_completo,
            mensagens=[
                {"role": "user", "content": pergunta},
            ],
            stream=False,
            max_tokens=120,
            temperatura=0.05,
            num_ctx=NUM_CTX_AUXILIAR,
            keep_alive=keep_alive,
            formato="json",
        )
        # Aceita também o envelope legado do SDK para manter integrações locais
        # que injetam um cliente compatível durante a migração de providers.
        if "message" in resultado:
            raw = resultado["message"]["content"].strip()
        else:
            if resultado["erro"]:
                raise RuntimeError(resultado["resposta"])
            raw = resultado["resposta"].strip()
        return _corrigir_inconsistencias(pergunta, _parse_resposta(raw))

    except Exception as e:
        logger.debug("Analisador falhou: %s", e)
        # Fallback barato: só ativa web quando a própria pergunta traz sinal de frescor.
        return IntencaoAnalisada(
            agente="generalista",
            precisa_web=_tem_sinal_atual(pergunta),
        )


def _parse_resposta(raw: str) -> IntencaoAnalisada:
    """Parseia o JSON da LLM com tolerância a erros."""
    # Com format="json", o output já deve ser JSON válido
    # Mas mantém parser robusto como fallback

    # Tenta parse direto primeiro
    try:
        data = json.loads(raw)
        return _validar_campos(data, raw)
    except json.JSONDecodeError:
        pass

    # Fallback: extrai JSON balanceado
    inicio = raw.find("{")
    if inicio == -1:
        return IntencaoAnalisada(raw=raw)

    depth = 0
    fim = -1
    for i in range(inicio, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                fim = i + 1
                break

    if fim == -1:
        return IntencaoAnalisada(raw=raw)

    json_str = raw[inicio:fim]

    try:
        data = json.loads(json_str)
        return _validar_campos(data, raw)
    except json.JSONDecodeError:
        return IntencaoAnalisada(raw=raw)


def _validar_campos(data: dict, raw: str) -> IntencaoAnalisada:
    """Valida e normaliza campos do JSON parseado."""
    agente = data.get("agente", "generalista")
    if agente not in _AGENTES_VALIDOS:
        agente = "generalista"

    precisa_web = bool(data.get("precisa_web", False))

    ferramenta = data.get("ferramenta")
    if ferramenta not in _FERRAMENTAS_VALIDAS:
        ferramenta = None

    parametros = data.get("parametros", {})
    if not isinstance(parametros, dict):
        parametros = {}

    return IntencaoAnalisada(
        agente=agente,
        precisa_web=precisa_web,
        ferramenta=ferramenta,
        parametros=parametros,
        raw=raw,
    )
