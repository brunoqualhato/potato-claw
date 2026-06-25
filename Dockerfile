# ─── Build stage: instala deps e compila extensões nativas ───
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# ─── Runtime stage: imagem mínima ───
FROM python:3.11-slim

WORKDIR /app

# Copia apenas os pacotes instalados (sem compiladores, headers, pip cache)
COPY --from=builder /install /usr/local

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
