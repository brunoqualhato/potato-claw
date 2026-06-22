"""
Busca web assíncrona para uso no pipeline nível 3.
Permite paralelizar busca web + embedding ChromaDB.
Fallback para versão síncrona se asyncio não estiver disponível.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

# Pool de threads leve — não consome RAM significativa
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="neuron-io")


async def executar_em_paralelo(*tarefas: tuple[Callable, tuple]) -> list[Any]:
    """
    Executa múltiplas funções síncronas em paralelo via thread pool.
    Ideal para I/O bound: busca web + embedding ao mesmo tempo.

    Uso:
        resultados = await executar_em_paralelo(
            (pesquisar_web, ("query", 3)),
            (gerar_embedding, ("texto",)),
        )
    """
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(_executor, lambda f=fn, a=args: f(*a))
        for fn, args in tarefas
    ]
    return await asyncio.gather(*futures, return_exceptions=True)


def paralelo_sync(*tarefas: tuple[Callable, tuple]) -> list[Any]:
    """
    Versão síncrona que cria event loop temporário.
    Para usar no pipeline que ainda é síncrono, sem reescrever tudo.

    Economia: busca web (~200-500ms) + ChromaDB query (~50ms)
    executam ao mesmo tempo em vez de sequencial.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Já dentro de async — fallback sequencial
            return [fn(*args) for fn, args in tarefas]
    except RuntimeError:
        pass

    return asyncio.run(executar_em_paralelo(*tarefas))
