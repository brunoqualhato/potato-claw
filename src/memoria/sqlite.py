"""
Camada 3 de memória: SQLite.
Histórico, resumos, contexto persistente e métricas.
Batch commits para reduzir I/O em disco lento.
"""

import atexit
import logging
import sqlite3
import statistics
from datetime import datetime

from src.core.config import MEMORIA_ARQUIVO

logger = logging.getLogger(__name__)


class Memoria:
    """Memória persistente com SQLite e batch commits."""

    def __init__(self, arquivo: str = MEMORIA_ARQUIVO):
        self.conn = sqlite3.connect(arquivo, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._pendente = False
        self._criar_tabelas()
        atexit.register(self._flush)

    def _flush(self):
        """Persiste transações pendentes."""
        if self._pendente:
            try:
                self.conn.commit()
                self._pendente = False
            except Exception as e:
                logger.debug("Erro no flush SQLite: %s", e)

    def _commit_batch(self):
        """Marca operação pendente sem forçar commit imediato."""
        self._pendente = True

    def flush(self):
        """Flush explícito — chamado no final de cada turno pelo executor."""
        self._flush()

    def _criar_tabelas(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS resumos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resumo TEXT NOT NULL,
                criado_em TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS contexto (
                chave TEXT PRIMARY KEY,
                valor TEXT NOT NULL,
                atualizado_em TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS historico (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                papel TEXT NOT NULL,
                conteudo TEXT NOT NULL,
                agente TEXT,
                nivel INTEGER DEFAULT 0,
                criado_em TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metricas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agente TEXT,
                nivel INTEGER,
                tempo_ms INTEGER,
                tokens_entrada INTEGER DEFAULT 0,
                tokens_saida INTEGER DEFAULT 0,
                fonte TEXT,
                criado_em TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def salvar_mensagem(self, papel: str, conteudo: str, agente: str | None = None, nivel: int = 0):
        self.conn.execute(
            "INSERT INTO historico (papel, conteudo, agente, nivel, criado_em) VALUES (?, ?, ?, ?, ?)",
            (papel, conteudo, agente, nivel, datetime.now().isoformat()),
        )
        self._commit_batch()

    def ultimas_mensagens(self, n: int = 3) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT papel, conteudo, agente FROM historico ORDER BY id DESC LIMIT ?",
            (n,),
        )
        rows = cursor.fetchall()
        return [
            {"role": r[0], "content": r[1], "agente": r[2]}
            for r in reversed(rows)
        ]

    def salvar_resumo(self, resumo: str):
        self.conn.execute(
            "INSERT INTO resumos (resumo, criado_em) VALUES (?, ?)",
            (resumo, datetime.now().isoformat()),
        )
        self._commit_batch()

    def ultimo_resumo(self) -> str | None:
        cursor = self.conn.execute(
            "SELECT resumo FROM resumos ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def definir_contexto(self, chave: str, valor: str):
        self.conn.execute(
            """INSERT OR REPLACE INTO contexto (chave, valor, atualizado_em)
               VALUES (?, ?, ?)""",
            (chave, valor, datetime.now().isoformat()),
        )
        self._commit_batch()

    def obter_contexto(self, chave: str) -> str | None:
        cursor = self.conn.execute(
            "SELECT valor FROM contexto WHERE chave = ?", (chave,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def total_mensagens(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM historico")
        return cursor.fetchone()[0]

    def salvar_metrica(self, agente: str, nivel: int, tempo_ms: int,
                       tokens_entrada: int = 0, tokens_saida: int = 0, fonte: str = ""):
        self.conn.execute(
            """INSERT INTO metricas (agente, nivel, tempo_ms, tokens_entrada, tokens_saida, fonte, criado_em)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (agente, nivel, tempo_ms, tokens_entrada, tokens_saida, fonte, datetime.now().isoformat()),
        )
        self._commit_batch()

    def metricas_resumo(self) -> dict:
        cursor = self.conn.execute(
            "SELECT nivel, tempo_ms, tokens_entrada, tokens_saida FROM metricas ORDER BY nivel, tempo_ms"
        )
        por_nivel: dict[int, list[tuple[int, int, int]]] = {}
        for nivel, tempo_ms, tokens_in, tokens_out in cursor.fetchall():
            por_nivel.setdefault(nivel, []).append((tempo_ms, tokens_in, tokens_out))

        resumo = {}
        for nivel, valores in por_nivel.items():
            tempos = [v[0] for v in valores]
            indice_p95 = max(0, min(len(tempos) - 1, round((len(tempos) - 1) * 0.95)))
            resumo[nivel] = {
                "total": len(valores),
                "avg_ms": round(statistics.fmean(tempos), 1),
                "p50_ms": round(statistics.median(tempos), 1),
                "p95_ms": tempos[indice_p95],
                "tokens_entrada": sum(v[1] for v in valores),
                "tokens_saida": sum(v[2] for v in valores),
            }
        return resumo

    def metricas_por_fonte(self) -> dict:
        cursor = self.conn.execute(
            """SELECT fonte, COUNT(*), AVG(tempo_ms)
               FROM metricas GROUP BY fonte ORDER BY COUNT(*) DESC"""
        )
        return {
            (fonte or "desconhecida"): {"total": total, "avg_ms": round(avg_ms, 1)}
            for fonte, total, avg_ms in cursor.fetchall()
        }

    def limpar_historico(self):
        self.conn.execute("DELETE FROM historico")
        self.conn.commit()  # Operação destrutiva: commit imediato

    def fechar(self):
        self._flush()
        self.conn.close()
