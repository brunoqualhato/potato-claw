# Changelog

## [Unreleased]

### Changed
- Pipeline nível 1 agora resolve ferramentas e cache antes do coordenador LLM
- Warm-up tornou-se seletivo e opt-in para reduzir pressão de RAM
- Memória vetorial separada por conversas, conhecimento, preferências, intenções e código
- API executa trabalho pesado fora do event loop e retorna agente, nível e fonte reais
- Docker inicia a API com dependências próprias

### Added
- Confirmação explícita para criação de arquivos e execução de comandos
- Proteção contra cache de consultas temporais
- Métricas de latência p50/p95 e tokens
- Smoke checks para projetos gerados e scaffold offline por templates

## [0.1.0] - 2026-06-23

### Added
- Sistema multiagente com 3 níveis de performance
- CLI interativa e modo batch
- API HTTP com FastAPI
- Memória em 3 camadas (cache JSON, SQLite, ChromaDB)
- Web RAG (search → fetch → extract)
- Loop autônomo de geração de código (`/projeto`)
- Template library para projetos estruturados
- Self-correction loop para modelos pequenos
- Perfis de modelos (ultra_leve, equilibrado, maximo)
- Suíte de testes com pytest
