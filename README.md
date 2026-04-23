# Churn Prediction

Pipeline completo de Machine Learning para previsão de churn de clientes de telecomunicações, incluindo EDA, modelagem com MLP (PyTorch), API REST (FastAPI) e suíte de testes abrangente.

O modelo identifica clientes com risco de cancelamento para que a equipe de retenção possa agir preventivamente.

**Stack:** Python 3.11+, PyTorch, scikit-learn, pandas, MLflow, FastAPI, pytest.

---

## Pré-requisitos

- Python **>= 3.11** (testado com 3.12)
- pip ou [uv](https://docs.astral.sh/uv/) (recomendado)

### Dependências principais

| Pacote | Versão | Finalidade |
|---|---|---|
| torch | >= 2.11 | Rede neural MLP |
| scikit-learn | >= 1.8 | Pré-processamento (ColumnTransformer, StandardScaler, OneHotEncoder) |
| pandas | >= 2.2, < 3 | Manipulação de dados |
| numpy | >= 1.26, < 2.5 | Operações numéricas |
| fastapi | >= 0.136 | API REST |
| uvicorn | >= 0.44 | Servidor ASGI |
| mlflow | >= 2.19 | Tracking de experimentos |
| matplotlib | >= 3.10 | Visualizações |
| seaborn | >= 0.13 | Visualizações estatísticas |
| openpyxl | >= 3.1 | Leitura de arquivos Excel |

### Dependências de teste (opcionais)

| Pacote | Finalidade |
|---|---|
| pytest >= 8.0 | Framework de testes |
| pytest-cov >= 5.0 | Cobertura de código |
| scipy >= 1.14 | Testes estatísticos (KS test) |
| pip-audit | Auditoria de vulnerabilidades (SEC-07) |

---

## Instalação

### Com pip

```bash
pip install -e .             # dependências do projeto
pip install -e ".[test]"     # inclui dependências de teste
pip install pip-audit        # opcional, para auditoria de segurança
```

### Com uv (recomendado)

```bash
uv sync                # instala dependências do projeto
uv sync --extra test   # inclui dependências de teste
```

---

## Executando o projeto

### 1. Treinar o modelo

O treinamento carrega o dataset, aplica pré-processamento (imputação, normalização, one-hot encoding), treina uma rede neural MLP e salva o modelo + preprocessor.

```bash
python -c "from src.model.train import run_training; run_training()"
```

Artefatos gerados:
- `src/saved_models/model.pt` — pesos do modelo PyTorch
- `src/saved_models/preprocessor.pkl` — pipeline de pré-processamento (sklearn)
- `docs/model_card.md` — documentação automática do modelo

### 2. Executar a API

A API REST permite fazer previsões de churn via HTTP.

```bash
uvicorn src.app.main:app --reload
```

A API estará disponível em `http://localhost:8000`.

#### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Health check — retorna `{"status": "ok"}` |
| POST | `/predict` | Previsão de churn para um cliente |
| GET | `/docs` | Documentação interativa (Swagger UI) |

#### Exemplo de requisição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "tenure": 24,
    "monthly_charges": 79.85,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check"
  }'
```

#### Exemplo de resposta

```json
{
  "probability": 0.72,
  "prediction": 1
}
```

- `probability` — probabilidade de churn (0.0 a 1.0)
- `prediction` — classificação binária (1 = churn, 0 = não churn, threshold: 0.5)

---

## Executando os Notebooks

Os notebooks Jupyter contêm a análise exploratória e os experimentos do projeto.

```bash
pip install jupyter          # se não tiver instalado
jupyter notebook notebooks/
```

| Notebook | Descrição |
|---|---|
| `01_eda.ipynb` | Análise Exploratória de Dados (EDA) — distribuições, correlações, insights |
| `02_baseline.ipynb` | Modelo baseline com Logistic Regression via scikit-learn |
| `03_experiments.ipynb` | Experimentos com MLP (PyTorch) e tracking via MLflow |

Os notebooks utilizam MLflow para tracking de métricas. Os resultados ficam salvos em `notebooks/mlruns/`.

---

## Executando os testes

```bash
pytest                  # executa todos os 95 testes
pytest -v               # modo verboso
pytest --cov            # com relatório de cobertura
```

### Executar por categoria

```bash
pytest tests/test_smoke.py          # Smoke Tests
pytest tests/test_data_quality.py   # Validação de dados
pytest tests/test_preprocessing.py  # Pré-processamento
pytest tests/test_model.py          # Modelo e validação
pytest tests/test_integration.py    # Integração (pipeline + MLflow)
pytest tests/test_blackbox.py       # Caixa-preta (cenários de negócio)
pytest tests/test_regression.py     # Regressão (prevenção de degradação)
pytest tests/test_robustness.py     # Robustez / caixa-branca
pytest tests/test_security.py       # Segurança, Pentest e LGPD
pytest tests/test_api.py            # API (FastAPI)
pytest tests/test_performance.py    # Performance (tempo e memória)
```

### Estrutura dos testes

```
tests/
  conftest.py              # fixtures compartilhadas (dataset, pipeline, dados sintéticos)
  test_smoke.py            # SM-01 a SM-05   — Smoke Tests (ambiente funcional)
  test_data_quality.py     # DQ-01 a DQ-09   — Validação de dados (schema, nulos, drift)
  test_preprocessing.py    # UT-01 a UT-10   — Unitários de pré-processamento
  test_model.py            # UT-11 a UT-14 + MV-01 a MV-10 — Modelo: treino, métricas, invariância
  test_integration.py      # IT-01 a IT-05   — Integração (pipeline end-to-end + MLflow)
  test_blackbox.py         # BB-01 a BB-06   — Caixa-preta (cenários reais de negócio)
  test_regression.py       # RG-01 a RG-04   — Regressão (thresholds mínimos de métricas)
  test_robustness.py       # WB-01 a WB-05   — Robustez (dados ausentes, edge cases)
  test_security.py         # SEC-01 a SEC-08  — Segurança (SQL injection, XSS, LGPD)
  test_api.py              # API-01 a API-07  — API REST (endpoints, validação, concorrência)
  test_performance.py      # PF-01 a PF-05   — Performance (tempo de treino/predição, memória)
```

### Detalhamento dos cenários de teste

#### Smoke Tests (`test_smoke.py`)

Verificações rápidas de sanidade para garantir que o ambiente está funcional antes de rodar a suíte completa.

| ID | Cenário | Objetivo |
|---|---|---|
| SM-01 | `main.py` executa sem erro | Valida que o ponto de entrada do projeto roda e imprime a mensagem esperada |
| SM-02 | Imports das dependências | Verifica que pandas, sklearn, mlflow e numpy estão instalados e importáveis |
| SM-03 | Dataset carrega corretamente | Confirma que o dataset bruto carrega com 7.043 linhas e 33 colunas |
| SM-04 | Pipeline fit/predict funciona | Garante que o pipeline completo (pré-processamento + modelo) treina e gera predições binárias (0/1) |
| SM-05 | MLflow tracking URI configurada | Verifica que o MLflow possui uma URI de tracking válida e não vazia |

#### Validação de Dados (`test_data_quality.py`)

Testes de qualidade e integridade do dataset bruto, garantindo que os dados atendem ao contrato esperado.

| ID | Cenário | Objetivo |
|---|---|---|
| DQ-01 | Schema de colunas | Valida que o dataset possui exatamente as 33 colunas esperadas, na ordem correta |
| DQ-02 | Contagem de linhas e unicidade | Confirma 7.043 registros e que não há CustomerIDs duplicados |
| DQ-03 | Nulos em Total Charges | Garante que a coluna Total Charges possui no máximo 11 valores nulos (esperados por clientes com tenure=0) |
| DQ-04 | Distribuição do target | Verifica que a taxa de churn está entre 25,5% e 27,5% (~26,5%), detectando alterações no balanceamento |
| DQ-05 | Categorias de contrato | Valida que Contract possui exatamente 3 valores: Month-to-month, One year, Two year |
| DQ-06 | Valores não negativos | Garante que Monthly Charges e Total Charges não contêm valores negativos |
| DQ-07 | Range de tenure | Verifica que Tenure Months está entre 0 e 72 meses |
| DQ-08 | Consistência CSV vs XLSX | Compara a contagem de linhas entre os arquivos CSV e XLSX para detectar divergências entre formatos |
| DQ-09 | Detecção de data drift (KS test) | Aplica o teste de Kolmogorov-Smirnov entre metades aleatórias do dataset para detectar drift nas distribuições numéricas |

#### Testes Unitários de Pré-processamento (`test_preprocessing.py`)

Validação isolada de cada etapa de transformação de dados.

| ID | Cenário | Objetivo |
|---|---|---|
| UT-01 | Conversão de Total Charges | Testa que `pd.to_numeric` converte strings numéricas e transforma espaços/vazios em NaN |
| UT-02 | Imputação numérica (mediana) | Verifica que SimpleImputer preenche NaNs numéricos com a mediana da coluna |
| UT-03 | Imputação categórica (moda) | Verifica que SimpleImputer preenche NaNs categóricos com o valor mais frequente |
| UT-04 | StandardScaler | Valida que a normalização resulta em média ≈ 0 e desvio padrão ≈ 1 |
| UT-05 | OneHotEncoder | Testa que categorias desconhecidas geram vetor de zeros (handle_unknown="ignore") e categorias conhecidas são codificadas corretamente |
| UT-06 | Remoção de colunas de leakage | Confirma que Churn Label, Churn Score e Churn Reason são removidos das features |
| UT-07 | Remoção de colunas irrelevantes | Confirma que CustomerID, Count, Lat Long, City, State e Zip Code são removidos |
| UT-08 | Substituição de espaços por NaN | Testa que strings contendo apenas espaço são convertidas para NaN |
| UT-09 | Agrupamento de tenure (pd.cut) | Valida que tenure é agrupado corretamente nos bins (0-12], (12-24], (24-48], (48-72] |
| UT-10 | Cálculo de avg_ticket | Verifica que avg_ticket = Total Charges / (Tenure Months + 1), sem divisão por zero |

#### Testes de Modelo (`test_model.py`)

Testes unitários do modelo (UT-11 a UT-14) e validação estatística com cross-validation (MV-01 a MV-10).

| ID | Cenário | Objetivo |
|---|---|---|
| UT-11 | Convergência do modelo | Verifica que a Logistic Regression converge sem ConvergenceWarning |
| UT-12 | Baseline com DummyClassifier | Confirma que o classificador "dummy" (majority class) tem F1 ≈ 0, Recall ≈ 0 e AUC ≈ 0.5 |
| UT-13 | Shape do ColumnTransformer | Valida que o pré-processador mantém o número de linhas e gera pelo menos n_numeric + OHE colunas |
| UT-14 | Pipeline aceita DataFrame bruto | Garante que o pipeline treinado aceita um DataFrame cru e retorna predições binárias |
| MV-01 | Recall mínimo por fold | Exige Recall ≥ 0.73 em cada fold da cross-validation (5 folds) |
| MV-02 | ROC-AUC mínimo e estabilidade | Exige AUC ≥ 0.84 em cada fold e desvio padrão < 0.02 entre folds |
| MV-03 | F1 mínimo por fold | Exige F1 ≥ 0.62 em cada fold |
| MV-04 | Supera o baseline | Valida que o modelo treinado supera o DummyClassifier em todas as métricas |
| MV-05 | Estabilidade do Recall | Verifica que o desvio padrão do Recall entre folds é < 0.03 |
| MV-06 | Detecção de overfitting | Garante que a diferença entre Recall de treino e teste é < 0.15 |
| MV-07 | Alinhamento de feature importance | Verifica que as top-20 features por importância incluem drivers identificados na EDA (Contract, Tenure, etc.) |
| MV-08 | Invariância de gênero (fairness) | Testa que alterar o gênero do cliente não muda a probabilidade média de churn em mais de 10% |
| MV-09 | Direcionalidade do tenure | Valida que o coeficiente de Tenure Months é negativo (mais tenure → menos churn) |
| MV-10 | Reprodutibilidade | Confirma que duas execuções de cross-validation com a mesma seed produzem resultados idênticos |

#### Testes de Integração (`test_integration.py`)

Validação do pipeline completo end-to-end e integração com MLflow.

| ID | Cenário | Objetivo |
|---|---|---|
| IT-01 | Pipeline end-to-end | Treina, prediz e valida que as predições são binárias e não degeneradas (nem 100% churn nem 100% não-churn) |
| IT-02 | Cross-validation completa | Verifica que cross_validate executa sem erros e retorna 5 folds com todas as métricas |
| IT-03 | Logging no MLflow | Testa que métricas, parâmetros e modelo são registrados corretamente em um experimento MLflow |
| IT-04 | Save/Load do modelo via MLflow | Salva o modelo treinado no MLflow, recarrega e valida que as predições são idênticas ao original |
| IT-05 | Formato do results.csv | Valida que o arquivo de resultados possui 4 colunas (f1, roc_auc, precision, recall), 5 linhas (folds) e valores entre 0 e 1 |

#### Testes Caixa-Preta (`test_blackbox.py`)

Cenários de negócio que validam se o modelo faz sentido do ponto de vista de domínio, sem conhecer a implementação interna.

| ID | Cenário | Objetivo |
|---|---|---|
| BB-01 | Perfil de alto risco de churn | Cliente mensal, fibra óptica, alta mensalidade, baixo tenure, sem suporte → probabilidade de churn > 50% |
| BB-02 | Perfil de baixo risco de churn | Cliente com contrato de 2 anos, DSL, baixa mensalidade, alto tenure, com suporte → predição = não churn |
| BB-03 | Idoso sem suporte técnico | Cliente idoso, sem Tech Support/Online Security, mensal, fibra óptica → probabilidade elevada (> 40%) |
| BB-04 | Fibra óptica + pagamento eletrônico | Combinação de Fiber optic + Electronic check + mensal → risco elevado (> 40%) |
| BB-05 | Distribuição em batch | Predição em batch de 100 clientes aleatórios deve ter taxa de churn entre 10% e 60% (intervalo plausível) |
| BB-06 | Determinismo | Mesma entrada executada duas vezes deve produzir exatamente as mesmas predições e probabilidades |

#### Testes de Regressão (`test_regression.py`)

Previnem degradação de performance ao longo do tempo, garantindo que mudanças no código não pioram o modelo.

| ID | Cenário | Objetivo |
|---|---|---|
| RG-01 | Métricas acima dos thresholds | Recall ≥ 0.73, AUC ≥ 0.84, F1 ≥ 0.62, Precision ≥ 0.50 (parametrizado para cada métrica) |
| RG-02 | Pipeline roda nas deps atuais | Verifica que o pipeline treina e prediz sem erros nas versões atuais das dependências |
| RG-03 | Nenhuma métrica abaixo do baseline | Garante que o modelo nunca performa abaixo do DummyClassifier em nenhuma métrica |
| RG-04 | Reprodutibilidade vs resultados salvos | Compara os resultados atuais da cross-validation com os salvos em `results.csv`, tolerância de 5% |

#### Testes de Robustez (`test_robustness.py`)

Validam o comportamento do pipeline em condições extremas e edge cases.

| ID | Cenário | Objetivo |
|---|---|---|
| WB-01 | Dataset vazio | Verifica que o pipeline levanta ValueError ou IndexError quando recebe um DataFrame vazio |
| WB-02 | Todas as categóricas são NaN | Testa que o pipeline consegue treinar e predizer mesmo quando todas as colunas categóricas são preenchidas com "missing" |
| WB-03 | Dataset com uma única linha | Avalia o comportamento com apenas 1 registro (aceita predição ou ValueError gracioso) |
| WB-04 | Coluna numérica constante | Verifica que o StandardScaler lida com colunas de variância zero sem gerar Inf ou NaN |
| WB-05 | StratifiedKFold vs cv=5 | Compara os resultados entre StratifiedKFold explícito e cv=5 (inteiro), garantindo diferença < 5% |

#### Testes de Segurança (`test_security.py`)

Testes de pentest, conformidade com LGPD e auditoria de dependências.

| ID | Cenário | Objetivo |
|---|---|---|
| SEC-01 | SQL Injection rejeitado | Envia payload com `'; DROP TABLE customers;--` e verifica retorno 400/422 |
| SEC-02 | XSS rejeitado | Envia payload com `<script>alert('xss')</script>` e valida que a resposta é JSON (não HTML) |
| SEC-03 | Acesso não autenticado | Envia requisição com body vazio e espera retorno 401, 403 ou 422 |
| SEC-04 | Path traversal bloqueado | Tenta acessar `/../../../etc/passwd` e verifica retorno 400/404/405 |
| SEC-05 | CustomerID fora das features (LGPD) | Garante que CustomerID não é utilizado como feature de entrada do modelo |
| SEC-06 | Colunas de localização fora das features (LGPD) | Garante que City, State, Zip Code e Lat Long não são features (proteção de dados sensíveis) |
| SEC-07 | Auditoria de vulnerabilidades (pip-audit) | Executa pip-audit e falha se houver vulnerabilidades CRITICAL ou HIGH nas dependências |
| SEC-08 | PII nos artefatos do MLflow | Varre os arquivos do MLflow buscando padrões de dados pessoais (CustomerID, Lat Long) |

#### Testes de API (`test_api.py`)

Validação dos endpoints da API REST (FastAPI).

| ID | Cenário | Objetivo |
|---|---|---|
| API-01 | Health check | Verifica que `GET /health` retorna status 200 com campo `status` |
| API-02 | Predição com payload válido | Envia dados completos de cliente e valida resposta com `prediction` (0/1) e `probability` (0.0–1.0) |
| API-03 | Campo obrigatório ausente | Envia payload incompleto e espera status 422 (Unprocessable Entity) |
| API-04 | Tipo de dado inválido | Envia string em campo numérico (`monthly_charges: "not_a_number"`) e espera status 422 |
| API-05 | Valores extremos | Envia `monthly_charges=999999.99` e `tenure=0` e valida que a API responde sem erro |
| API-06 | Requisições concorrentes | Dispara 10 requisições simultâneas (ThreadPool) e valida que todas retornam status 200 |
| API-07 | Campos extras ignorados | Envia campos não esperados pelo schema e verifica que a API os ignora, retornando 200 |

#### Testes de Performance (`test_performance.py`)

Garantem que o modelo e a API atendem a requisitos não-funcionais de tempo e memória.

| ID | Cenário | Objetivo |
|---|---|---|
| PF-01 | Tempo de treinamento | Treinamento completo do pipeline deve levar < 10 segundos |
| PF-02 | Tempo de predição unitária | Predição de um único registro deve levar < 100ms (mediana de 5 execuções, com warm-up) |
| PF-03 | Tempo de predição em batch | Predição do dataset completo deve levar < 2 segundos |
| PF-04 | Uso de memória no treinamento | Pico de memória durante o treinamento deve ser < 500MB |
| PF-05 | Tempo de startup da API | Import do módulo FastAPI deve levar < 5 segundos |

---

## Estrutura do projeto

```
churn-prediction/
├── main.py                          # ponto de entrada CLI
├── pyproject.toml                   # metadados, dependências e config do pytest
├── uv.lock                          # lockfile de dependências (uv)
│
├── data/raw/                        # dataset Telco Customer Churn (.csv e .xlsx)
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Análise Exploratória de Dados
│   ├── 02_baseline.ipynb            # Modelo baseline (Logistic Regression)
│   ├── 03_experiments.ipynb         # Experimentos MLP + MLflow
│   └── mlruns/                      # resultados do MLflow (tracking local)
│
├── src/
│   ├── app/                         # API FastAPI
│   │   ├── main.py                  #   aplicação FastAPI
│   │   ├── routes.py                #   endpoints (/health, /predict)
│   │   ├── schemas.py               #   schema Pydantic de entrada
│   │   └── model_loader.py          #   carregamento do modelo para inferência
│   │
│   ├── model/                       # modelos ML
│   │   ├── mlp.py                   #   arquitetura MLP (PyTorch)
│   │   ├── train.py                 #   pipeline de treinamento
│   │   ├── evaluate.py              #   avaliação de métricas
│   │   ├── predict.py               #   predição standalone
│   │   └── prepare_data.py          #   carregamento e pré-processamento dos dados
│   │
│   ├── pipeline/
│   │   └── preprocessing.py         #   pré-processamento de entrada para inferência
│   │
│   ├── services/
│   │   └── model_service.py         #   serviço de predição (lazy load do modelo)
│   │
│   ├── config/
│   │   └── settings.py              #   configurações (paths, threshold, epochs)
│   │
│   ├── saved_models/                # artefatos treinados
│   │   ├── model.pt                 #   pesos do modelo MLP
│   │   └── preprocessor.pkl         #   pipeline de pré-processamento
│   │
│   ├── utils/
│   │   ├── model_card.py            #   geração automática do model card
│   │   └── preprocessing.py         #   utilitários de pré-processamento
│   │
│   └── features/
│       └── build_features.py        #   engenharia de features
│
├── tests/                           # suíte de testes (95 testes)
│
├── docs/
│   ├── ML_CANVAS.md                 # ML Canvas (problema, métricas, stakeholders)
│   ├── model_card.md                # Model Card (gerado automaticamente)
│   ├── monitoring.md                # estratégia de monitoramento
│   └── ROTEIRO_VIDEO_STAR.md        # roteiro de apresentação
│
└── mlruns/                          # tracking do MLflow (raiz)
```

---

## Métricas do modelo

| Métrica | Valor |
|---|---|
| Accuracy | 73.0% |
| Precision | 49.5% |
| Recall | 76.5% |
| F1-score | 0.60 |
| ROC-AUC | 0.81 |

O modelo prioriza **Recall** para minimizar falsos negativos (clientes que cancelam sem serem identificados).

---

## Documentação adicional

- [`docs/ML_CANVAS.md`](docs/ML_CANVAS.md) — Definição do problema, métricas de sucesso e stakeholders
- [`docs/model_card.md`](docs/model_card.md) — Model Card com métricas, limitações e uso em produção
- [`docs/monitoring.md`](docs/monitoring.md) — Estratégia de monitoramento do modelo
