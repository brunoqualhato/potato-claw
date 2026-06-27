"""Limites compartilhados dos recursos pesados do runtime local."""

import threading

from src.core.config import MAX_INFERENCIAS_OLLAMA

# Chat, embeddings e warmup competem pela mesma RAM/GPU no Ollama.
SEMAFORO_OLLAMA = threading.BoundedSemaphore(MAX_INFERENCIAS_OLLAMA)
