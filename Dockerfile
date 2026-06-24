FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
