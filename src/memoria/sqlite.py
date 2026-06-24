"""
Camada 3 de memória: SQLite.
Histórico, resumos, contexto persistente e métricas.

Suporta isolamento por sessão (canal, pessoa) para multi-user via `sessao_ativa`.
`sessao_ativa = ""` mantém o comportamento global (single-user/CLI), retrocompatível.
"""

import logging
import sqlite3
import statistics
from datetime import datetime

from src.core.config import MEMORIA_ARQUIVO

logger = logging.getLogger(__name__)


class Memoria:
    """Memória persistente com SQLite."""

    def __init__(self, arquivo: str = MEMORIA_ARQUIVO):
        # check_same_thread=False: tanto o servidor de canais (runtime, via
        # asyncio.to_thread) quanto a API executam o pipeline em worker threads,
        # entao a conexao e usada por threads diferentes do pool. O acesso e
        # serializado (uma mensagem por vez), entao nao ha concorrencia real. WAL ajuda.
        self.conn = sqlite3.connect(arquivo, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        # Sessao ativa para isolar historico/resumos por (canal, pessoa). "" = global.
        self.sessao_ativa = ""
        self._criar_tabelas()
        self._migrar()

    def _criar_tabelas(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS resumos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resumo TEXT NOT NULL,
                sessao TEXT DEFAULT '',
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
                sessao TEXT DEFAULT '',
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

    def _migrar(self):
        """Adiciona a coluna `sessao` em bancos antigos (retrocompativel)."""
        for tabela in ("historico", "resumos"):
            try:
                self.conn.execute(f"ALTER TABLE {tabela} ADD COLUMN sessao TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # coluna ja existe
        self.conn.commit()

    def salvar_mensagem(self, papel: str, conteudo: str, agente: str | None = None, nivel: int = 0):
        self.conn.execute(
            "INSERT INTO historico (papel, conteudo, agente, nivel, sessao, criado_em) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (papel, conteudo, agente, nivel, self.sessao_ativa, datetime.now().isoformat()),
        )
        self.conn.commit()

    def ultimas_mensagens(self, n: int = 3) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT papel, conteudo, agente FROM historico WHERE sessao = ? "
            "ORDER BY id DESC LIMIT ?",
            (self.sessao_ativa, n),
        )
        rows = cursor.fetchall()
        return [
            {"role": r[0], "content": r[1], "agente": r[2]}
            for r in reversed(rows)
        ]

    def salvar_resumo(self, resumo: str):
        self.conn.execute(
            "INSERT INTO resumos (resumo, sessao, criado_em) VALUES (?, ?, ?)",
            (resumo, self.sessao_ativa, datetime.now().isoformat()),
        )
        self.conn.commit()

    def ultimo_resumo(self) -> str | None:
        cursor = self.conn.execute(
            "SELECT resumo FROM resumos WHERE sessao = ? ORDER BY id DESC LIMIT 1",
            (self.sessao_ativa,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def definir_contexto(self, chave: str, valor: str):
        self.conn.execute(
            """INSERT OR REPLACE INTO contexto (chave, valor, atualizado_em)
               VALUES (?, ?, ?)""",
            (chave, valor, datetime.now().isoformat()),
        )
        self.conn.commit()

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
        self.conn.commit()

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
        self.conn.commit()

    def fechar(self):
        self.conn.close()
