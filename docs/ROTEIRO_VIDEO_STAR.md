# Roteiro de Vídeo (5 min) — Modelo STAR — Churn Prediction

**Duração total:** 5 minutos  
**Formato:** Screencast com narração (tela do Cursor/VS Code + voz)  
**Estrutura:** STAR (Situation, Task, Action, Result)

---

## Preparação antes de gravar

Deixe as seguintes abas já abertas no editor, nesta ordem:

1. `docs/ML_CANVAS.md`
2. `notebooks/01_eda.ipynb`
3. `pyproject.toml`
4. `src/model/prepare_data.py`
5. `src/model/mlp.py`
6. `src/model/train.py`
7. `src/model/threshold_tuning.py`
8. `src/app/routes.py`
9. `src/app/middleware.py`
10. `src/services/model_service.py`
11. `src/app/schemas.py`
12. `src/utils/logger.py`
13. `src/monitoring/drift_monitor.py`
14. `src/monitoring/business_monitor.py`
15. `tests/` (explorer lateral aberto nesta pasta)
16. `src/utils/model_card.py`
17. `Dockerfile`
18. `docker-compose.yml`
19. `infra/prometheus.yml`
20. `docs/architecture.md`
21. `docs/monitoring.md`
22. `makefile`
23. `README.md`

Dica: use `Ctrl+Tab` para navegar entre abas durante a gravação. Não digite código ao vivo — apenas navegue entre arquivos pré-abertos.

---

## S — SITUATION (Situação) · 0:00 – 1:00

> **Objetivo da seção:** Contextualizar o problema de negócio para que o avaliador entenda *por que* esse projeto existe.

### Tela: `docs/ML_CANVAS.md` (seção 1 — Definição do Problema)

**Fala sugerida (adapte com suas palavras):**

> "Olá! Neste vídeo vou apresentar o projeto de Churn Prediction que desenvolvi como parte do Tech Challenge.
>
> O cenário é o seguinte: uma operadora de telecomunicações está enfrentando uma **taxa de churn de 26,5%** — ou seja, mais de 1 em cada 4 clientes cancela o serviço todo mês. Isso gera um impacto enorme no faturamento, já que o custo de aquisição de um novo cliente é muito maior do que o custo de reter um existente.
>
> O dataset utilizado é o **Telco Customer Churn**, com 7.043 registros de clientes e variáveis como tipo de contrato, tempo de permanência, cobranças mensais, serviços contratados e dados demográficos."

### Tela: `notebooks/01_eda.ipynb` (role até os gráficos principais)

> "Na análise exploratória, identifiquei padrões muito claros. Por exemplo:
>
> - Clientes com **contrato mensal** têm 42,7% de churn, contra apenas 2,8% nos contratos bienais.
> - Clientes **sem segurança online** cancelam em 41,8% dos casos.
> - E os churners pagam, em média, **R$74 por mês** contra R$61 dos que ficam.
>
> Esses insights guiaram tanto a seleção de features quanto a estratégia de modelagem."

**Tempo:** ~60 segundos

---

## T — TASK (Tarefa) · 1:00 – 1:45

> **Objetivo da seção:** Dizer claramente *o que* precisava ser entregue e quais decisões técnicas foram tomadas.

### Tela: Explorer lateral mostrando a árvore do projeto (pastas `src/`, `tests/`, `notebooks/`, `docs/`)

**Fala sugerida:**

> "A tarefa era construir um **pipeline completo de Machine Learning** — da ingestão e tratamento dos dados até o deploy containerizado, com **monitoramento e observabilidade** de ponta a ponta.
>
> A **métrica técnica principal** que escolhi foi o **Recall**. A razão é simples: o custo de *não* identificar um cliente que vai cancelar — um falso negativo — é muito maior do que acionar a equipe de retenção para um cliente que não ia cancelar. É melhor errar pra mais do que deixar passar.
>
> Além do modelo em si, o projeto precisava incluir boas práticas de ML Engineering:
> - **Versionamento de experimentos** com MLflow
> - Uma **suíte de testes abrangente** com 95 testes cobrindo desde smoke tests até segurança e LGPD
> - **Documentação** com ML Canvas, Model Card automático, arquitetura e plano de monitoramento
> - Uma **API REST** com FastAPI, middleware de latência e logging estruturado
> - **Monitoramento** com detecção de data drift e log de predições para métricas de negócio
> - **Containerização** com Docker, Docker Compose, Prometheus e Grafana"

### Tela: `pyproject.toml` (scroll rápido mostrando dependências)

> "As restrições incluíam conformidade com a **LGPD** no uso dos dados, o modelo precisava ser **acionável semanalmente** pelo time de CRM, e havia 11 valores nulos no campo Total Charges que precisavam de tratamento. Além disso, o sistema precisava ter **rastreabilidade** — cada requisição identificada e com latência medida — e capacidade de detectar **data drift** para saber quando retreinar o modelo."

**Tempo:** ~45 segundos

---

## A — ACTION (Ação) · 1:45 – 4:30

> **Objetivo da seção:** Detalhar *como* você construiu cada peça. É a parte mais técnica — mostre domínio, mas não leia código linha a linha. Destaque decisões.

---

### A.1 — Pré-processamento de Dados · 1:45 – 2:15

### Tela: `src/model/prepare_data.py` (foco nas linhas 37–55, o ColumnTransformer)

**Fala sugerida:**

> "A primeira etapa foi o pré-processamento. Implementei um **ColumnTransformer** do scikit-learn com dois sub-pipelines:
>
> - Para **colunas numéricas**: imputação pela mediana e normalização com StandardScaler.
> - Para **colunas categóricas**: imputação pela moda e OneHotEncoder.
>
> Um ponto fundamental: o **split dos dados é feito antes do fit do preprocessor**. Isso evita *data leakage* — o preprocessor aprende somente com os dados de treino, e depois é aplicado nos dados de teste. É um erro comum que compromete a validação do modelo se não for tratado."

### Tela: scroll para as linhas 58–64 (train_test_split estratificado + fit_transform)

> "O split usa **estratificação pelo target**, garantindo que a proporção de churn (26,5%) se mantenha igual em treino e teste."

**Tempo:** ~30 segundos

---

### A.2 — Arquitetura e Treinamento do Modelo · 2:15 – 2:55

### Tela: `src/model/mlp.py` (arquivo inteiro visível)

**Fala sugerida:**

> "Para o modelo, optei por uma **rede neural MLP em PyTorch** com quatro camadas: 128, 64, 32 e 1 neurônio de saída.
>
> Três decisões de design importantes aqui:
>
> 1. **BatchNorm** entre as camadas — estabiliza o treinamento e permite learning rates maiores.
> 2. **Dropout progressivo** (0.4, 0.3 e 0.2) — reduz overfitting de forma gradual ao longo da rede.
> 3. A saída é um **logit único**, e não uma sigmoid — porque a loss function que uso, `BCEWithLogitsLoss`, já aplica a sigmoid internamente com mais estabilidade numérica."

### Tela: `src/model/train.py` (foco no training loop com DataLoader e early stopping)

> "No treinamento, implementei três técnicas fundamentais:
>
> - **Batching com DataLoader** — os dados são processados em mini-batches de 64 amostras, o que estabiliza o gradiente e permite treinar com datasets maiores.
> - **Early stopping com paciência de 10 épocas** — o treinamento para automaticamente se a loss não melhorar por 10 épocas consecutivas, evitando overfitting.
> - E o **pos_weight** na loss function, que penaliza mais os erros na classe minoritária, compensando o desbalanceamento de 73,5% vs 26,5%."

### Tela: `src/model/threshold_tuning.py`

> "Além disso, implementei um **threshold tuning automático**. Em vez de usar o threshold fixo de 0.5, o sistema faz uma busca de 0.1 a 0.9 em passos de 0.05, escolhendo o threshold que maximiza o **F1-score**. No modelo atual, o threshold ótimo ficou em torno de **0.3** — o que aumenta o recall sem sacrificar demais a precision. Esse threshold é salvo junto com o modelo e usado automaticamente em produção."

**Tempo:** ~40 segundos

---

### A.3 — API, Middleware e Serviço de Predição · 2:55 – 3:25

### Tela: `src/app/routes.py`

**Fala sugerida:**

> "A API REST foi construída com **FastAPI** e expõe dois endpoints:
> - `GET /health` — para monitoramento e health checks.
> - `POST /predict` — recebe os dados de um cliente e retorna a probabilidade de churn e a classificação binária."

### Tela: `src/app/middleware.py`

> "Implementei um **middleware de latência** que intercepta todas as requisições. Cada request recebe um **UUID único** (`X-Request-ID`) para rastreabilidade, e o tempo de processamento é medido e retornado no header `X-Process-Time-ms`. Tudo é registrado com **logging estruturado** — método, path, status code e latência — facilitando o diagnóstico em produção."

### Tela: `src/utils/logger.py`

> "O logging usa um logger centralizado chamado `churn_api`, com formato estruturado que inclui timestamp, nível, nome do logger e mensagem. Isso padroniza os logs da API inteira."

### Tela: `src/services/model_service.py` (foco no lazy loading)

> "O serviço de predição usa **lazy loading** — o modelo e o preprocessor são carregados do disco apenas na primeira requisição e ficam em cache na memória. Além disso, cada predição é **logada automaticamente** em CSV para monitoramento posterior de métricas de negócio."

### Tela: `src/app/schemas.py`

> "A validação de entrada usa **Pydantic**, com campos tipados e aliases. E o preprocessor garante que a transformação em produção seja **idêntica** à do treinamento — colunas faltantes são preenchidas automaticamente, mantendo a consistência do pipeline."

**Tempo:** ~35 segundos

---

### A.4 — Testes e Qualidade · 3:25 – 3:50

### Tela: Explorer lateral na pasta `tests/` (mostrando todos os 12 arquivos de teste)

**Fala sugerida:**

> "A suíte de testes tem **12 módulos** com **95 testes** organizados em categorias:
>
> - **Smoke tests** — verificam que imports e estrutura básica funcionam.
> - **Data quality** — validam schema, tipos e valores nulos do dataset.
> - **Unitários e pré-processamento** — testam transformações e comportamento do modelo.
> - **Integração** — validam o pipeline completo com MLflow.
> - **Caixa-preta** — simulam cenários reais de negócio.
> - **Regressão** — previnem degradação de performance.
> - **Robustez** — verificam resiliência a dados faltantes e outliers.
> - **Segurança** — validam conformidade com LGPD.
> - **API** — testam endpoints, validação e respostas HTTP.
> - **Performance** — medem tempo de inferência e uso de memória."

### Tela: `src/utils/model_card.py` (scroll rápido)

> "Também implementei geração automática de **Model Card** — um documento que registra métricas, parâmetros, limitações e instruções de uso do modelo, seguindo boas práticas de ML responsável."

### Tela: `makefile`

> "Para facilitar o fluxo de desenvolvimento, criei um **Makefile** com targets para as operações mais comuns: `make api` para subir o servidor, `make train` para treinar o modelo, `make test` para rodar a suíte completa, `make lint` para verificação de código com Ruff, e `make audit` para auditoria de segurança das dependências."

**Tempo:** ~30 segundos

---

### A.5 — Monitoramento e Observabilidade · 3:50 – 4:10

### Tela: `src/monitoring/drift_monitor.py`

**Fala sugerida:**

> "Implementei um sistema de **monitoramento** com dois módulos. O primeiro é o **drift monitor**, que compara a distribuição dos dados atuais com uma baseline salva durante o treinamento. Se a média de alguma feature numérica desvia mais de 30% do baseline, o sistema emite um alerta — indicando que pode ser hora de retreinar o modelo."

### Tela: `src/monitoring/business_monitor.py`

> "O segundo é o **monitor de negócio**. Cada predição da API é registrada em CSV com customer ID, probabilidade e classificação. Quando rótulos reais ficam disponíveis, o sistema calcula automaticamente **precision, recall e F1** — permitindo medir a performance do modelo em produção."

**Tempo:** ~20 segundos

---

### A.6 — Containerização e Infraestrutura · 4:10 – 4:30

### Tela: `Dockerfile`

**Fala sugerida:**

> "Para garantir que o sistema rode em qualquer ambiente, implementei a **containerização com Docker**. O Dockerfile usa `python:3.11-slim` como base, instala as dependências e expõe a API na porta 8000."

### Tela: `docker-compose.yml`

> "O **Docker Compose** orquestra quatro serviços:
> - A **API de churn** — o container principal com FastAPI.
> - O **MLflow** — para visualização dos experimentos de treinamento.
> - O **Prometheus** — coletando métricas da aplicação a cada 15 segundos.
> - E o **Grafana** — para dashboards de monitoramento em tempo real.
>
> Com um único `docker compose up`, toda a stack sobe pronta para uso."

### Tela: `infra/prometheus.yml` (scroll rápido)

> "O Prometheus está configurado para scrapear a API automaticamente, usando o service discovery do Docker Compose."

**Tempo:** ~20 segundos

---

## R — RESULT (Resultado) · 4:30 – 5:30

> **Objetivo da seção:** Mostrar os resultados concretos e fechar com reflexão + próximos passos.

---

### R.1 — Métricas do Modelo · 4:30 – 4:50

### Tela: `docs/model_card.md` (seção Métricas)

**Fala sugerida:**

> "O modelo MLP alcançou métricas sólidas nos dados de teste. As cinco métricas avaliadas são: **Accuracy**, **Precision**, **Recall** — que é a métrica principal —, **F1-score** e **ROC-AUC**.
>
> Um diferencial importante: o **threshold não é fixo em 0.5**. Implementei um **threshold tuning automático** que busca o valor que maximiza o F1-score. No modelo atual, o threshold ótimo ficou em **~0.3** — o que aumenta significativamente o recall, identificando mais clientes em risco de churn. Esse threshold é salvo junto com o modelo e carregado automaticamente em produção."

**Tempo:** ~20 segundos

---

### R.2 — Entregáveis e Valor de Negócio · 4:50 – 5:10

### Tela: `README.md` (visão geral)

**Fala sugerida:**

> "O entregável é um **sistema completo, containerizado e pronto para produção**:
>
> - Modelo com **early stopping**, **batching** e **threshold tuning** automático
> - **API REST** com FastAPI, middleware de latência e logging estruturado
> - **95 testes** em 12 módulos cobrindo desde smoke tests até LGPD
> - **Monitoramento** com detecção de data drift e métricas de negócio
> - **Stack Docker** completa: API + MLflow + Prometheus + Grafana com um único `docker compose up`
> - Documentação completa: **ML Canvas**, **Model Card** automático, **arquitetura** e **plano de monitoramento**
>
> Com esse sistema, a equipe de retenção pode receber semanalmente a lista de clientes em risco e agir preventivamente — atacando diretamente a taxa de churn de 26,5%."

**Tempo:** ~20 segundos

---

### R.3 — Aprendizados e Próximos Passos · 5:10 – 5:30

### Tela: `docs/ML_CANVAS.md` (voltar ao início como fechamento circular)

**Fala sugerida:**

> "Os principais aprendizados foram:
>
> - A importância de **evitar data leakage** — split antes do fit do preprocessor.
> - O impacto do **threshold tuning** — um threshold de 0.3 em vez de 0.5 faz diferença real no recall.
> - Como **early stopping** evita overfitting sem precisar definir o número de épocas manualmente.
> - O valor de **containerizar desde o início** — Docker Compose unifica API, MLflow e monitoramento em um comando.
>
> Como próximos passos, planejo:
> - **CI/CD automatizado** com GitHub Actions para rodar testes e lint a cada commit.
> - Integrar o drift monitor ao **pipeline de retreinamento automático**.
> - Adicionar **exporter de métricas** para enriquecer os dashboards do Grafana.
> - E experimentar outros algoritmos como **XGBoost** para comparação de performance.
>
> Obrigado pela atenção! Estou à disposição para dúvidas."

**Tempo:** ~20 segundos

---

## Resumo de Tempos

| Seção | Intervalo | Duração |
|-------|-----------|---------|
| **S — Situation** | 0:00 – 1:00 | 60s |
| **T — Task** | 1:00 – 1:45 | 45s |
| **A — Action** (6 sub-seções) | 1:45 – 4:30 | 165s |
| **R — Result** (3 sub-seções) | 4:30 – 5:30 | 60s |
| **Total** | | **~5:30** |

> **Nota:** O roteiro ficou ~30 segundos acima dos 5 min. Ajuste o ritmo da narração nas seções A.3 e A.6, ou consolide A.5 e A.6 em uma única seção para caber no tempo.

---

## Dicas de Gravação

- **Ensaie pelo menos 2x** antes de gravar. O roteiro está detalhado para servir de guia, mas adapte as frases ao seu jeito natural de falar.
- **Não leia código linha a linha** — aponte para os trechos relevantes e explique a intenção por trás das decisões.
- **Pause brevemente** entre seções STAR para dar ritmo e permitir que o avaliador acompanhe a mudança de contexto.
- **Troque a aba** visível no editor a cada sub-seção para manter o vídeo visualmente dinâmico.
- **Fale com confiança** na seção de Result — é o momento de mostrar orgulho do que foi construído.
- **Encerre com firmeza** — não deixe o final "morrer". A última frase deve soar como ponto final, não reticências.
