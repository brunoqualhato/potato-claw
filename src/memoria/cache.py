"""
Camada 1 de memória: Cache exato (JSON) com LRU.
Hash da pergunta → resposta instantânea.
Eviction policy: máximo 500 entradas, remove as menos usadas.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from src.core.config import CACHE_ARQUIVO, CACHE_HABILITADO

logger = logging.getLogger(__name__)

CACHE_MAX_ENTRADAS = 500  # Limite para não crescer indefinidamente


class Cache:
    """Cache de respostas com LRU eviction para controlar RAM."""

    def __init__(self, arquivo: str = CACHE_ARQUIVO):
        self.arquivo = Path(arquivo)
        self.dados: dict[str, dict] = {}
        self._dirty = False
        self._carregar()

    def _carregar(self):
        if self.arquivo.exists():
            try:
                self.dados = json.loads(self.arquivo.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Cache corrompido, iniciando vazio: %s", e)
                self.dados = {}

    def _salvar(self):
        """Escrita atômica: write → rename para evitar corrupção."""
        try:
            self.arquivo.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.arquivo.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self.dados, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(self.arquivo)  # Atômico no mesmo filesystem
            self._dirty = False
        except OSError as e:
            logger.warning("Erro ao salvar cache: %s", e)

    def _evict_se_necessario(self):
        """Remove entradas LRU se exceder tamanho máximo."""
        if len(self.dados) <= CACHE_MAX_ENTRADAS:
            return
        # Ordena por ultimo_uso e remove as mais antigas
        ordenado = sorted(
            self.dados.items(),
            key=lambda kv: kv[1].get("ultimo_uso", ""),
        )
        remover = len(self.dados) - CACHE_MAX_ENTRADAS
        for chave, _ in ordenado[:remover]:
            del self.dados[chave]

    @staticmethod
    def _hash(texto: str) -> str:
        return hashlib.sha256(texto.strip().lower().encode()).hexdigest()[:16]

    @staticmethod
    def _consulta_base(pergunta: str) -> str:
        if "\x1f" in pergunta:
            _, pergunta = pergunta.split("\x1f", 1)
        if ":" in pergunta:
            return pergunta.split(":", 1)[1].strip().lower()
        return pergunta.strip().lower()

    @classmethod
    def _nao_cachear_consulta(cls, pergunta: str) -> bool:
        base = cls._consulta_base(pergunta)
        if not base:
            return True
        termos_temporais = {
            "hoje", "agora", "atual", "atuais", "recente", "recentes",
            "último", "ultima", "última", "latest", "cotação", "preço",
            "clima", "temperatura", "placar", "resultado", "versão", "versao",
            "release", "lançamento", "lancamento", "notícia", "noticia",
            "dólar", "dolar", "euro", "bitcoin", "câmbio", "cambio",
        }
        if any(termo in base for termo in termos_temporais):
            return True
        tokens = base.split()
        if len(tokens) <= 2:
            genericas = {
                "oi", "olá", "ola", "hello", "hey", "e ai", "e aí",
                "ok", "blz", "valeu", "obrigado", "obg", "sim", "não", "nao",
            }
            if base in genericas:
                return True
        return False

    def buscar(self, pergunta: str) -> str | None:
        if not CACHE_HABILITADO:
            return None
        if self._nao_cachear_consulta(pergunta):
            return None
        chave = self._hash(pergunta)
        entry = self.dados.get(chave)
        if entry:
            entry["hits"] = entry.get("hits", 0) + 1
            entry["ultimo_uso"] = datetime.now().isoformat()
            self._dirty = True
            self._salvar()
            return entry["resposta"]
        return None

    def salvar(self, pergunta: str, resposta: str, agente: str = ""):
        if not CACHE_HABILITADO:
            return
        if self._nao_cachear_consulta(pergunta):
            return
        chave = self._hash(pergunta)
        self.dados[chave] = {
            "resposta": resposta,
            "agente": agente,
            "hits": 1,
            "criado_em": datetime.now().isoformat(),
            "ultimo_uso": datetime.now().isoformat(),
        }
        self._evict_se_necessario()
        self._salvar()

    def limpar(self):
        self.dados = {}
        self._salvar()

    def estatisticas(self) -> dict:
        total = len(self.dados)
        hits_total = sum(e.get("hits", 0) for e in self.dados.values())
        return {"entradas": total, "max": CACHE_MAX_ENTRADAS, "hits_total": hits_total}
