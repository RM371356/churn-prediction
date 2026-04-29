.PHONY: help run api train test lint format security audit clean install

help:
	@echo "Comandos disponíveis:"
	@echo "  make run       -> sobe a API FastAPI"
	@echo "  make api       -> alias para run"
	@echo "  make train     -> treina o modelo"
	@echo "  make test      -> roda os testes"
	@echo "  make lint      -> verifica lint com Ruff"
	@echo "  make format    -> corrige lint + formata código"
	@echo "  make security  -> roda testes de segurança"
	@echo "  make audit     -> verifica vulnerabilidades com pip-audit"
	@echo "  make install   -> instala dependências"
	@echo "  make clean     -> limpa cache e arquivos temporários"

run:
	uvicorn src.app.main:app --reload

api:
	uvicorn src.app.main:app --reload

train:
	python -m src.model.train

test:
	pytest -v

lint:
	ruff check .

format:
	ruff check . --fix
	ruff format .

security:
	pytest tests/test_security.py -v

audit:
	pip-audit

install:
	pip install -e .[dev]

clean:
	python -c "import shutil, os; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', '__pycache__']]"