# Churn Prediction

Projeto de Machine Learning para predição de churn de clientes de telecomunicações.
O modelo identifica clientes com risco de cancelamento para que a equipe de retenção
possa agir preventivamente.

**Stack:** Python 3.12+, scikit-learn, pandas, MLflow, FastAPI (planejado).

## Pré-requisitos

- Python **>= 3.12**
- [uv](https://docs.astral.sh/uv/) (recomendado) ou pip

## Instalação

### Com uv (recomendado)

```bash
uv sync                # instala dependências do projeto
uv sync --extra test   # inclui dependências de teste
```

### Com pip

```bash
pip install .            # dependências do projeto
pip install ".[test]"    # inclui dependências de teste (pytest, pytest-cov, scipy)
```

## Executar os testes

```bash
pytest              # executa todos os testes
pytest -v           # modo verboso (mostra cada teste individualmente)
pytest --cov        # com relatório de cobertura
```

### Executar uma categoria específica

```bash
pytest tests/test_smoke.py          # apenas smoke tests
pytest tests/test_data_quality.py   # apenas validação de dados
pytest tests/test_model.py          # apenas testes de modelo
```

## Estrutura dos testes

```
tests/
  conftest.py            # fixtures compartilhadas (dataset, pipeline, dados sintéticos)
  test_smoke.py          # SM-01 a SM-05  — Smoke Tests
  test_data_quality.py   # DQ-01 a DQ-09  — Validação de dados
  test_preprocessing.py  # UT-01 a UT-10  — Testes unitários de pré-processamento
  test_model.py          # UT-11 a UT-14, MV-01 a MV-10 — Testes de modelo e validação
  test_integration.py    # IT-01 a IT-05  — Testes de integração (pipeline + MLflow)
  test_blackbox.py       # BB-01 a BB-06  — Testes caixa-preta (cenários de negócio)
  test_regression.py     # RG-01 a RG-04  — Testes de regressão (prevenção de degradação)
  test_robustness.py     # WB-01 a WB-05  — Testes de robustez / caixa-branca
  test_security.py       # SEC-01 a SEC-08 — Segurança, Pentest e LGPD
  test_api.py            # API-01 a API-07 — Testes de API (ativados quando FastAPI for implementada)
  test_performance.py    # PF-01 a PF-05  — Testes de performance (tempo e memória)
```

## Estrutura do projeto

```
churn-prediction-01/
  main.py                 # ponto de entrada CLI
  pyproject.toml          # metadados e dependências
  data/raw/               # dataset Telco Customer Churn
  notebooks/              # EDA, baseline e experimentos
  src/api/                # API FastAPI (em desenvolvimento)
  src/models/             # modelos ML
  src/pipelines/          # pipeline de pré-processamento
  docs/                   # documentação (ML Canvas)
  tests/                  # suíte de testes
```
