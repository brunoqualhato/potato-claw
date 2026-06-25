"""
Camada 2 de memória: Memória Semântica (ChromaDB).
Busca por similaridade vetorial usando embeddings locais.
"""

import hashlib
import logging
import re
from datetime import datetime

import chromadb
import ollama as ollama_client

from src.core.config import (
    CHROMADB_COLLECTION,
    CHROMADB_DIR,
    CHROMADB_THRESHOLD,
    CHROMADB_TOP_K,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class MemoriaSemantica:
    """
    Memória vetorial com ChromaDB (Singleton).
    Armazena pares pergunta+resposta e busca por similaridade.
    Usa embeddings locais do Ollama (nomic-embed-text).
    """

    _instancia: "MemoriaSemantica | None" = None

    def __new__(cls):
        """Singleton — reutiliza a mesma conexão ChromaDB."""
        if cls._instancia is None:
            cls._instancia = super().__new__(cls)
            cls._instancia._inicializado = False
        return cls._instancia

    def __init__(self):
        if self._inicializado:
            return
        self.client = chromadb.PersistentClient(path=CHROMADB_DIR)
        nomes = {
            "conversa": CHROMADB_COLLECTION,
            "conhecimento": f"{CHROMADB_COLLECTION}_conhecimento",
            "preferencia": f"{CHROMADB_COLLECTION}_preferencias",
            "intencao_classificada": f"{CHROMADB_COLLECTION}_intencoes",
            "codigo_gerado": f"{CHROMADB_COLLECTION}_codigo",
            "arquitetura": f"{CHROMADB_COLLECTION}_codigo",
        }
        self.collections = {
            tipo: self.client.get_or_create_collection(
                name=nome,
                metadata={"hnsw:space": "cosine"},
            )
            for tipo, nome in nomes.items()
        }
        # Compatibilidade com integrações existentes: conversa é a coleção padrão.
        self.collection = self.collections["conversa"]
        self._migrar_tipos_legados()
        self._embedding_disponivel = None
        self._inicializado = True

    @classmethod
    def resetar_instancia(cls):
        """Permite resetar o singleton (útil para testes)."""
        cls._instancia = None

    def _verificar_embedding(self) -> bool:
        """Verifica se o modelo de embedding está disponível."""
        if self._embedding_disponivel is not None:
            return self._embedding_disponivel
        try:
            ollama_client.embeddings(model=EMBEDDING_MODEL, prompt="teste")
            self._embedding_disponivel = True
        except Exception as e:
            logger.debug("Modelo de embedding '%s' indisponível: %s", EMBEDDING_MODEL, e)
            self._embedding_disponivel = False
        return self._embedding_disponivel

    def _migrar_tipos_legados(self):
        """Move documentos tipados que versões antigas gravavam na coleção padrão."""
        try:
            total = self.collection.count()
            if total == 0:
                return
            dados = self.collection.get(include=["documents", "metadatas", "embeddings"])
            for indice, doc_id in enumerate(dados["ids"]):
                metadata = (dados.get("metadatas") or [])[indice] or {}
                tipo = metadata.get("tipo", "conversa")
                destino = self.collections.get(tipo)
                if tipo == "conversa" or destino is None or destino.name == self.collection.name:
                    continue
                destino.upsert(
                    ids=[doc_id],
                    documents=[dados["documents"][indice]],
                    embeddings=[dados["embeddings"][indice]],
                    metadatas=[metadata],
                )
                self.collection.delete(ids=[doc_id])
        except Exception as e:
            logger.debug("Migração de memória legada ignorada: %s", e)

    def _gerar_embedding(self, texto: str) -> list[float]:
        """Gera embedding usando Ollama."""
        response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=texto)
        return response["embedding"]

    @staticmethod
    def _tokenizar(texto: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9_\-]{2,}", texto.lower()))

    def _score_lexical(self, pergunta: str, documento: str) -> float:
        """Jaccard simples para re-ranking híbrido com baixo custo."""
        q = self._tokenizar(pergunta)
        d = self._tokenizar(documento)
        if not q or not d:
            return 0.0
        intersecao = len(q & d)
        uniao = len(q | d)
        if uniao == 0:
            return 0.0
        return intersecao / uniao

    def buscar_similar(
        self,
        pergunta: str,
        top_k: int = CHROMADB_TOP_K,
        tipos: tuple[str, ...] | None = None,
        sessao: str | None = None,
    ) -> list[dict]:
        """
        Busca documentos similares.
        Retorna lista de {conteudo, similaridade, metadata}.
        """
        if not self._verificar_embedding():
            return []

        try:
            embedding = self._gerar_embedding(pergunta)
            documentos = []
            tipos_busca = tipos or ("conversa", "conhecimento", "preferencia")
            colecoes_vistas: set[str] = set()
            for tipo in tipos_busca:
                collection = self.collections.get(tipo)
                if collection is None or collection.name in colecoes_vistas:
                    continue
                colecoes_vistas.add(collection.name)
                total = collection.count()
                if total == 0:
                    continue

                query_kwargs = {
                    "query_embeddings": [embedding],
                    "n_results": min(top_k, total),
                    "include": ["documents", "metadatas", "distances"],
                }
                if tipo == "conversa" and sessao is not None:
                    if sessao:
                        query_kwargs["where"] = {"sessao": sessao}
                    else:
                        # Bancos antigos não possuem o metadado `sessao`.
                        # Consulta os candidatos e filtra abaixo sem expor
                        # conversas de sessões nomeadas ao modo global/CLI.
                        query_kwargs["n_results"] = total

                results = collection.query(**query_kwargs)

                for i, doc in enumerate(results["documents"][0]):
                    distancia = results["distances"][0][i]
                    similaridade = 1 - (distancia / 2)

                    if similaridade >= CHROMADB_THRESHOLD:
                        score_lexical = self._score_lexical(pergunta, doc)
                        score_hibrido = (similaridade * 0.82) + (score_lexical * 0.18)
                        metadata = results["metadatas"][0][i] or {}
                        tipo_doc = metadata.get("tipo")
                        if tipo_doc and tipo_doc not in tipos_busca:
                            continue
                        if (
                            tipo == "conversa"
                            and sessao is not None
                            and metadata.get("sessao", "") != sessao
                        ):
                            continue
                        documentos.append({
                            "conteudo": doc,
                            "similaridade": similaridade,
                            "score_lexical": score_lexical,
                            "score_hibrido": score_hibrido,
                            "metadata": metadata,
                        })

            documentos.sort(key=lambda d: d.get("score_hibrido", d["similaridade"]), reverse=True)
            return documentos[:top_k]

        except Exception as e:
            logger.warning("Erro ao buscar similar no ChromaDB: %s", e)
            return []

    def adicionar(self, pergunta: str, resposta: str, agente: str = "", metadata: dict = None):
        """Adiciona par pergunta+resposta ao ChromaDB com deduplicação."""
        if not self._verificar_embedding():
            return

        try:
            documento = f"Pergunta: {pergunta}\nResposta: {resposta}"

            # Deduplicação: hash baseado no conteúdo (sem timestamp)
            sessao = (metadata or {}).get("sessao", "")
            identidade = f"{sessao}\0{pergunta.strip().lower()}"
            doc_id = hashlib.sha256(identidade.encode()).hexdigest()[:16]

            # Se já existe um doc com mesmo ID, verifica se vale atualizar
            existing = self.collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                doc_existente = existing["documents"][0] if existing["documents"] else ""
                if len(documento) <= len(doc_existente):
                    return  # Não sobrescreve com resposta pior

            embedding = self._gerar_embedding(pergunta)

            meta = {
                "agente": agente,
                "tipo": "conversa",
                "sessao": sessao,
                "criado_em": datetime.now().isoformat(),
            }
            if metadata:
                meta.update(metadata)

            self.collections["conversa"].upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[documento],
                metadatas=[meta],
            )
        except Exception as e:
            logger.warning("Erro ao adicionar ao ChromaDB: %s", e)

    def adicionar_conhecimento(self, texto: str, fonte: str = "", tipo: str = "conhecimento"):
        """Adiciona conhecimento avulso (docs, notas, etc) com deduplicação."""
        if not self._verificar_embedding():
            return

        try:
            doc_id = hashlib.sha256(texto.strip().lower()[:200].encode()).hexdigest()[:16]

            collection = self.collections.get(tipo, self.collections["conhecimento"])
            existing = collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                return  # Já existe

            embedding = self._gerar_embedding(texto)

            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[texto],
                metadatas=[{
                    "tipo": tipo,
                    "fonte": fonte,
                    "criado_em": datetime.now().isoformat(),
                }],
            )
        except Exception as e:
            logger.warning("Erro ao adicionar conhecimento ao ChromaDB: %s", e)

    def total_documentos(self) -> int:
        return sum(
            collection.count()
            for collection in {collection.name: collection for collection in self.collections.values()}.values()
        )

    def estatisticas(self) -> dict:
        return {
            "documentos": self.total_documentos(),
            "por_colecao": {
                collection.name: collection.count()
                for collection in {
                    collection.name: collection for collection in self.collections.values()
                }.values()
            },
            "embedding_model": EMBEDDING_MODEL,
            "disponivel": self._verificar_embedding(),
        }
