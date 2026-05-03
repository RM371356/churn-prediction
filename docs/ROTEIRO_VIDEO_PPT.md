# Roteiro de Vídeo (5 min) — Modelo STAR — Churn Prediction
## Versão com Apresentação PPT

**Duração total:** 5 minutos  
**Formato:** Screencast com narração (apresentação PPT em tela cheia + voz)  
**Estrutura:** STAR (Situation, Task, Action, Result)  
**Arquivo PPT:** `docs/apresentacao_churn_prediction.pptx`

---

## Preparação antes de gravar

1. Abra o arquivo `docs/apresentacao_churn_prediction.pptx` no PowerPoint
2. Coloque em modo **Apresentação de Slides** (F5)
3. Teste a passagem de slides com as setas do teclado
4. Deixe o cursor do mouse invisível (Ctrl+H no modo apresentação)

Dica: Ensaie a passagem de slides com o roteiro pelo menos 2x. Os slides já contêm os pontos principais — use-os como guia visual enquanto narra.

---

## Slide 1 — Capa

> **Fala sugerida:**
>
> "Olá! Neste vídeo vou apresentar o projeto de Churn Prediction que desenvolvemos como parte do Tech Challenge. Vou seguir a estrutura STAR — Situação, Tarefa, Ação e Resultado."

**Ação:** Avance para o próximo slide.

**Tempo:** ~10 segundos

---

## Slide 2 — Agenda

> "A apresentação está organizada em quatro seções: primeiro o contexto do problema, depois o que precisava ser entregue, em seguida como construímos cada componente, e por fim os resultados alcançados."

**Ação:** Avance para o slide de seção S.

**Tempo:** ~10 segundos

---

## S — SITUATION (Situação) · 0:00 – 1:00

---

### Slide 3 — Divisória "S — SITUATION"

> Pause brevemente (2-3 segundos) para o avaliador registrar a mudança de seção.

**Ação:** Avance.

---

### Slide 4 — Problema de Negócio

> **Fala sugerida:**
>
> "O cenário é o seguinte: uma operadora de telecomunicações está enfrentando uma **taxa de churn de 26,5%** — ou seja, mais de 1 em cada 4 clientes cancela o serviço todo mês. Isso gera um impacto enorme no faturamento, já que o custo de aquisição de um novo cliente é muito maior do que o custo de reter um existente.
>
> O dataset utilizado é o **Telco Customer Churn**, com 7.043 registros e variáveis como tipo de contrato, tempo de permanência, cobranças mensais e serviços contratados."

**Ação:** Avance.

**Tempo:** ~20 segundos

---

### Slide 5 — Insights da EDA

> **Fala sugerida:**
>
> "Na análise exploratória, identificamos padrões muito claros:
>
> - Clientes com **contrato mensal** têm 42,7% de churn, contra apenas 2,8% nos contratos bienais.
> - Clientes **sem segurança online** cancelam em 41,8% dos casos.
> - E os churners pagam, em média, **R$74 por mês** contra R$61 dos que ficam.
>
> Esses insights guiaram tanto a seleção de features quanto a estratégia de modelagem."

**Ação:** Avance.

**Tempo:** ~20 segundos

---

### Slide 6 — Tabela de Variáveis Relevantes

> "Aqui vocês podem ver as 7 variáveis que se mostraram mais relevantes na análise. Destacamos o contrato, tenure e monthly charges como os drivers mais fortes de churn."

**Ação:** Avance para a seção T.

**Tempo:** ~10 segundos

---

## T — TASK (Tarefa) · 1:00 – 1:45

---

### Slide 7 — Divisória "T — TASK"

> Pause brevemente.

**Ação:** Avance.

---

### Slide 8 — O Que Precisava Ser Entregue

> **Fala sugerida:**
>
> "A tarefa era construir um **pipeline completo de Machine Learning** — da ingestão e tratamento dos dados até o deploy containerizado, com monitoramento de ponta a ponta.
>
> A **métrica principal** que escolhemos foi o **Recall**. A razão é simples: o custo de *não* identificar um cliente que vai cancelar é muito maior do que acionar a equipe de retenção desnecessariamente.
>
> O projeto inclui: versionamento com MLflow, 95 testes, documentação completa, API REST com FastAPI, monitoramento com drift detection, e containerização com Docker Compose."

**Ação:** Avance.

**Tempo:** ~25 segundos

---

### Slide 9 — Restrições e Decisões

> "As restrições incluíam conformidade com a **LGPD**, modelo acionável semanalmente pelo CRM, rastreabilidade com UUID por requisição, e capacidade de detectar **data drift** para saber quando retreinar."

**Ação:** Avance para a seção A.

**Tempo:** ~15 segundos

---

## A — ACTION (Ação) · 1:45 – 4:30

---

### Slide 10 — Divisória "A — ACTION"

> Pause brevemente.

**Ação:** Avance.

---

### A.1 — Pré-processamento de Dados · 1:45 – 2:15

---

### Slide 11 — Pré-processamento de Dados

> **Fala sugerida:**
>
> "A primeira etapa foi o pré-processamento. Implementamos um **ColumnTransformer** com dois sub-pipelines:
>
> - Para **numéricas**: imputação pela mediana e StandardScaler.
> - Para **categóricas**: imputação pela moda e OneHotEncoder.
>
> Ponto fundamental: o **split dos dados é feito antes do fit do preprocessor** — evitando data leakage. E o split usa estratificação pelo target."

**Ação:** Avance para o slide de código.

**Tempo:** ~20 segundos

---

### Slide 12 — Código do Pipeline

> "Aqui vocês podem ver o código. Notem que o `fit_transform` é chamado apenas no `X_train`, e o `X_test` recebe apenas `transform` — garantindo que não há vazamento de informação."

**Ação:** Avance.

**Tempo:** ~10 segundos

---

### A.2 — Arquitetura e Treinamento · 2:15 – 2:55

---

### Slide 13 — Arquitetura MLP

> **Fala sugerida:**
>
> "Para o modelo, optamos por uma **rede neural MLP em PyTorch** com quatro camadas: 128, 64, 32 e 1 neurônio.
>
> Três decisões importantes:
> 1. **BatchNorm** — estabiliza o treinamento.
> 2. **Dropout progressivo** (0.4, 0.3, 0.2) — reduz overfitting gradualmente.
> 3. Saída é **logit único** — BCEWithLogitsLoss aplica sigmoid internamente com mais estabilidade."

**Ação:** Avance.

**Tempo:** ~15 segundos

---

### Slide 14 — Código da MLP

> "Aqui a arquitetura completa. Note o padrão: Linear → ReLU → BatchNorm → Dropout, repetido com dimensões decrescentes."

**Ação:** Avance.

**Tempo:** ~8 segundos

---

### Slide 15 — Threshold Tuning

> "No treinamento, usei **early stopping** com paciência de 10 épocas e **pos_weight** para compensar o desbalanceamento.
>
> Além disso, implementei **threshold tuning automático** — busca de 0.1 a 0.9 maximizando F1. O threshold ótimo ficou em **0.30**, o que aumenta significativamente o recall. Esse valor é salvo junto com o modelo."

**Ação:** Avance.

**Tempo:** ~17 segundos

---

### A.3 — API e Serviço de Predição · 2:55 – 3:25

---

### Slide 16 — API REST

> **Fala sugerida:**
>
> "A API foi construída com **FastAPI** e expõe dois endpoints: health check e predição.
>
> O middleware gera UUID único por requisição e mede latência. O serviço usa **lazy loading** — modelo carregado na primeira chamada. E cada predição é logada em CSV para monitoramento posterior."

**Ação:** Avance.

**Tempo:** ~15 segundos

---

### Slide 17 — Código do Endpoint

> "Aqui o endpoint de predição e um exemplo de requisição e resposta. A validação de entrada é feita com Pydantic."

**Ação:** Avance.

**Tempo:** ~8 segundos

---

### A.4 — Testes e Qualidade · 3:25 – 3:50

---

### Slide 18 — Suíte de Testes

> **Fala sugerida:**
>
> "A suíte tem **12 módulos** com **95 testes** organizados em categorias: desde smoke tests até segurança e LGPD, passando por caixa-preta, regressão e performance."

**Ação:** Avance.

**Tempo:** ~10 segundos

---

### Slide 19 — Tabela de Testes

> "Nesta tabela, cada módulo com seus IDs e foco. Destaco os testes de segurança que validam SQL injection, XSS e conformidade LGPD, e os de performance que garantem inferência em menos de 100ms."

**Ação:** Avance.

**Tempo:** ~12 segundos

---

### A.5 — Monitoramento · 3:50 – 4:10

---

### Slide 20 — Monitoramento e Observabilidade

> **Fala sugerida:**
>
> "Implementei dois monitores. O **drift monitor** compara distribuições atuais com baseline e alerta se desviarem mais de 30%. O **business monitor** registra cada predição e calcula métricas reais quando rótulos ficam disponíveis."

**Ação:** Avance.

**Tempo:** ~15 segundos

---

### A.6 — Containerização · 4:10 – 4:30

---

### Slide 21 — Docker e Infraestrutura

> **Fala sugerida:**
>
> "Para containerização, o **Docker Compose** orquestra quatro serviços: a API, MLflow, Prometheus e Grafana. Com um único `docker compose up`, toda a stack sobe pronta para uso."

**Ação:** Avance.

**Tempo:** ~10 segundos

---

### Slide 22 — Arquitetura dos Containers

> "O fluxo vai do cliente para a API, que consulta o modelo e retorna a predição. Em paralelo, Prometheus coleta métricas e Grafana exibe dashboards."

**Ação:** Avance para a seção R.

**Tempo:** ~8 segundos

---

## R — RESULT (Resultado) · 4:30 – 5:30

---

### Slide 23 — Divisória "R — RESULT"

> Pause brevemente.

**Ação:** Avance.

---

### R.1 — Métricas · 4:30 – 4:50

---

### Slide 24 — Métricas do Modelo

> **Fala sugerida:**
>
> "O modelo alcançou **Recall de 72,2%** — nossa métrica principal — com threshold otimizado em 0.30. O ROC-AUC ficou em 0.827, mostrando boa capacidade de discriminação. O diferencial é que o threshold não é fixo: ele é tuned automaticamente e salvo com o modelo."

**Ação:** Avance.

**Tempo:** ~18 segundos

---

### R.2 — Entregáveis · 4:50 – 5:10

---

### Slide 25 — Entregáveis e Valor de Negócio

> **Fala sugerida:**
>
> "O entregável é um **sistema completo, containerizado e pronto para produção**: modelo com early stopping e threshold tuning, API REST, 95 testes, monitoramento com drift detection, stack Docker completa, e documentação detalhada.
>
> Com esse sistema, a equipe de retenção pode receber semanalmente a lista de clientes em risco e agir preventivamente."

**Ação:** Avance.

**Tempo:** ~18 segundos

---

### R.3 — Aprendizados · 5:10 – 5:30

---

### Slide 26 — Aprendizados e Próximos Passos

> **Fala sugerida:**
>
> "Os principais aprendizados: evitar data leakage com split antes do fit, o impacto real do threshold tuning, como early stopping evita overfitting automaticamente, e o valor de containerizar desde o início.
>
> Como próximos passos: CI/CD com GitHub Actions, retreinamento automático via drift monitor, e experimentar XGBoost para comparação."

**Ação:** Avance para o slide final.

**Tempo:** ~18 segundos

---

### Slide 27 — Encerramento

> "Obrigado pela atenção! Estou à disposição para dúvidas."

**Tempo:** ~5 segundos

---

## Resumo de Tempos

| Seção | Slides | Intervalo | Duração |
|-------|--------|-----------|---------|
| Capa + Agenda | 1–2 | 0:00 – 0:20 | 20s |
| **S — Situation** | 3–6 | 0:20 – 1:00 | 40s |
| **T — Task** | 7–9 | 1:00 – 1:45 | 45s |
| **A — Action** (6 sub-seções) | 10–22 | 1:45 – 4:30 | 165s |
| **R — Result** (3 sub-seções) | 23–27 | 4:30 – 5:30 | 60s |
| **Total** | 27 slides | | **~5:30** |

> **Nota:** O roteiro ficou ~30 segundos acima dos 5 min. Ajuste acelerando levemente nas seções A.3 e A.6 (slides de código podem ser passados mais rápido).

---

## Mapeamento Slides ↔ Conteúdo

| Slide | Conteúdo | Arquivo de Referência |
|-------|----------|----------------------|
| 1 | Capa | — |
| 2 | Agenda STAR | — |
| 3 | Divisória S | — |
| 4 | Problema de negócio | docs/ML_CANVAS.md |
| 5 | Insights EDA | notebooks/01_eda.ipynb |
| 6 | Tabela de variáveis | docs/ML_CANVAS.md |
| 7 | Divisória T | — |
| 8 | Entregáveis necessários | — |
| 9 | Restrições e stack | pyproject.toml |
| 10 | Divisória A | — |
| 11 | Pré-processamento | src/model/prepare_data.py |
| 12 | Código preprocessing | src/model/prepare_data.py |
| 13 | Arquitetura MLP | src/model/mlp.py |
| 14 | Código MLP | src/model/mlp.py |
| 15 | Threshold tuning | src/model/threshold_tuning.py |
| 16 | API REST | src/app/routes.py, middleware.py |
| 17 | Código endpoint | src/app/routes.py |
| 18 | Suíte de testes | tests/ |
| 19 | Tabela de testes | tests/ |
| 20 | Monitoramento | src/monitoring/ |
| 21 | Docker/Infra | docker-compose.yml |
| 22 | Arquitetura containers | docker-compose.yml |
| 23 | Divisória R | — |
| 24 | Métricas | docs/model_card.md |
| 25 | Entregáveis | README.md |
| 26 | Aprendizados | docs/ML_CANVAS.md |
| 27 | Encerramento | — |

---

## Dicas de Gravação

- **Use a apresentação em tela cheia** — mais profissional que navegar entre abas do editor.
- **Não leia os slides** — eles são suporte visual. Fale com naturalidade usando suas palavras.
- **Pause nas divisórias de seção** — dá ritmo e permite que o avaliador acompanhe.
- **Slides de código:** não explique linha a linha. Aponte para as decisões (ex: "notem o fit_transform apenas no train").
- **Fale com confiança na seção R** — é o momento de mostrar orgulho do que foi construído.
- **Encerre com firmeza** — a última frase deve soar como ponto final.
- **Ensaie 2-3x** com cronômetro antes de gravar.

---

## Vantagens desta versão vs. screencast com editor

| Aspecto | Screencast (editor) | PPT |
|---------|-------------------|-----|
| Profissionalismo visual | Médio | Alto |
| Risco de problemas técnicos | Alto (abas, zoom, linter) | Baixo |
| Controle de tempo | Difícil | Fácil (1 slide = X seg) |
| Foco do avaliador | Disperso (muita info) | Direcionado |
| Preparação | Muitas abas abertas | 1 arquivo |
| Código mostrado | Todo (pode confundir) | Trechos selecionados |

---

## Como Regenerar o PPT

Se precisar atualizar os slides (ex: após retreinar o modelo com novas métricas):

```bash
py scripts/generate_ppt.py
```

O script `scripts/generate_ppt.py` gera automaticamente o arquivo `docs/apresentacao_churn_prediction.pptx` com todos os 27 slides.
