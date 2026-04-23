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
7. `src/app/routes.py`
8. `src/services/model_service.py`
9. `src/app/schemas.py`
10. `tests/` (explorer lateral aberto nesta pasta)
11. `src/utils/model_card.py`
12. `README.md`

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

> "A tarefa era construir um **pipeline completo de Machine Learning** — da ingestão e tratamento dos dados até a exposição de uma API de predição pronta para produção.
>
> A **métrica técnica principal** que escolhi foi o **Recall**. A razão é simples: o custo de *não* identificar um cliente que vai cancelar — um falso negativo — é muito maior do que acionar a equipe de retenção para um cliente que não ia cancelar. É melhor errar pra mais do que deixar passar.
>
> Além do modelo em si, o projeto precisava incluir boas práticas de ML Engineering:
> - **Versionamento de experimentos** com MLflow
> - Uma **suíte de testes abrangente** cobrindo desde smoke tests até segurança e LGPD
> - **Documentação** com ML Canvas e Model Card automático
> - Uma **API REST** com FastAPI para consumo do modelo"

### Tela: `pyproject.toml` (scroll rápido mostrando dependências)

> "As restrições incluíam conformidade com a **LGPD** no uso dos dados, o modelo precisava ser **acionável semanalmente** pelo time de CRM, e havia 11 valores nulos no campo Total Charges que precisavam de tratamento."

**Tempo:** ~45 segundos

---

## A — ACTION (Ação) · 1:45 – 3:45

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

### A.2 — Arquitetura do Modelo · 2:15 – 2:45

### Tela: `src/model/mlp.py` (arquivo inteiro visível)

**Fala sugerida:**

> "Para o modelo, optei por uma **rede neural MLP em PyTorch** com quatro camadas: 128, 64, 32 e 1 neurônio de saída.
>
> Três decisões de design importantes aqui:
>
> 1. **BatchNorm** entre as camadas — estabiliza o treinamento e permite learning rates maiores.
> 2. **Dropout progressivo** (0.4, 0.3 e 0.2) — reduz overfitting de forma gradual ao longo da rede.
> 3. A saída é um **logit único**, e não uma sigmoid — porque a loss function que uso, `BCEWithLogitsLoss`, já aplica a sigmoid internamente com mais estabilidade numérica."

### Tela: `src/model/train.py` (foco nas linhas 35–47, pos_weight e criterion)

> "No treinamento, calculei o **pos_weight** — a razão entre exemplos negativos e positivos — e passei para a loss function. Isso faz o modelo penalizar mais os erros na classe minoritária (churn), compensando o **desbalanceamento de 73,5% vs 26,5%**."

**Tempo:** ~30 segundos

---

### A.3 — API e Serviço de Predição · 2:45 – 3:15

### Tela: `src/app/routes.py`

**Fala sugerida:**

> "A API REST foi construída com **FastAPI** e expõe dois endpoints:
> - `GET /health` — para monitoramento e health checks.
> - `POST /predict` — recebe os dados de um cliente e retorna a probabilidade de churn e a classificação binária."

### Tela: `src/services/model_service.py` (foco nas linhas 13–28, lazy loading)

> "O serviço de predição usa **lazy loading** — o modelo e o preprocessor são carregados do disco apenas na primeira requisição e ficam em cache na memória. Isso evita recarregar os artefatos a cada chamada."

### Tela: `src/app/schemas.py`

> "A validação de entrada usa **Pydantic**, com campos tipados e aliases. E o preprocessor garante que a transformação em produção seja **idêntica** à do treinamento — colunas faltantes são preenchidas automaticamente, mantendo a consistência do pipeline."

**Tempo:** ~30 segundos

---

### A.4 — Testes e Qualidade · 3:15 – 3:45

### Tela: Explorer lateral na pasta `tests/` (mostrando todos os arquivos)

**Fala sugerida:**

> "A suíte de testes tem **10 categorias** com mais de 60 testes:
>
> - **Smoke tests** — verificam que imports e estrutura básica funcionam.
> - **Data quality** — validam schema, tipos e valores nulos do dataset.
> - **Unitários** — testam pré-processamento e comportamento do modelo.
> - **Integração** — validam o pipeline completo com MLflow.
> - **Caixa-preta** — simulam cenários reais de negócio.
> - **Regressão** — previnem degradação de performance.
> - **Robustez** — verificam resiliência a dados faltantes e outliers.
> - **Segurança** — validam conformidade com LGPD.
> - **Performance** — medem tempo de inferência e uso de memória."

### Tela: `src/utils/model_card.py` (scroll rápido)

> "Também implementei geração automática de **Model Card** — um documento que registra métricas, parâmetros, limitações e instruções de uso do modelo, seguindo boas práticas de ML responsável."

**Tempo:** ~30 segundos

---

## R — RESULT (Resultado) · 3:45 – 5:00

> **Objetivo da seção:** Mostrar os resultados concretos e fechar com reflexão + próximos passos.

---

### R.1 — Métricas do Modelo · 3:45 – 4:15

### Tela: Saída do treinamento no terminal ou `docs/model_card.md` (seção Métricas)

**Fala sugerida:**

> "O modelo MLP alcançou métricas sólidas nos dados de teste. As cinco métricas avaliadas são:
>
> - **Accuracy** — acurácia geral do modelo.
> - **Precision** — dos que o modelo classificou como churn, quantos realmente eram.
> - **Recall** — dos clientes que realmente cancelaram, quantos o modelo identificou. Essa é a **métrica principal**.
> - **F1-score** — média harmônica entre precision e recall.
> - **ROC-AUC** — capacidade de discriminação entre as classes.
>
> O threshold de 0.5 é configurável. Se o negócio quiser ser mais conservador — identificar mais clientes em risco — basta reduzir o threshold para aumentar o recall."

**Tempo:** ~30 segundos

---

### R.2 — Entregáveis e Valor de Negócio · 4:15 – 4:40

### Tela: `README.md` (visão geral)

**Fala sugerida:**

> "O entregável é um **sistema completo e pronto para produção**:
>
> - Modelo treinado e serializado (`model.pt`)
> - Preprocessor versionado (`preprocessor.pkl`)
> - API REST com FastAPI
> - Suíte de testes com mais de 60 casos
> - Documentação com **ML Canvas** e **Model Card** automático
>
> Com esse sistema, a equipe de retenção pode receber semanalmente a lista de clientes em risco e agir preventivamente — atacando diretamente a taxa de churn de 26,5%. O pipeline é totalmente reprodutível: basta rodar `uv sync` e executar o treinamento."

**Tempo:** ~25 segundos

---

### R.3 — Aprendizados e Próximos Passos · 4:40 – 5:00

### Tela: `docs/ML_CANVAS.md` (voltar ao início como fechamento circular)

**Fala sugerida:**

> "Os principais aprendizados foram:
>
> - A importância de **evitar data leakage** no pré-processamento — fazer split antes do fit.
> - O impacto do **desbalanceamento de classes** e como `pos_weight` na loss function resolve isso.
> - Como uma suíte de testes robusta dá **confiança para iterar rapidamente**.
>
> Como próximos passos, planejo:
> - **Containerização com Docker** para deploy.
> - **CI/CD automatizado** para rodar testes a cada commit.
> - **Monitoramento de data drift** em produção.
> - E experimentar outros algoritmos como **XGBoost** para comparação.
>
> Obrigado pela atenção! Estou à disposição para dúvidas."

**Tempo:** ~20 segundos

---

## Resumo de Tempos

| Seção | Intervalo | Duração |
|-------|-----------|---------|
| **S — Situation** | 0:00 – 1:00 | 60s |
| **T — Task** | 1:00 – 1:45 | 45s |
| **A — Action** (4 sub-seções) | 1:45 – 3:45 | 120s |
| **R — Result** (3 sub-seções) | 3:45 – 5:00 | 75s |
| **Total** | | **5:00** |

---

## Dicas de Gravação

- **Ensaie pelo menos 2x** antes de gravar. O roteiro está detalhado para servir de guia, mas adapte as frases ao seu jeito natural de falar.
- **Não leia código linha a linha** — aponte para os trechos relevantes e explique a intenção por trás das decisões.
- **Pause brevemente** entre seções STAR para dar ritmo e permitir que o avaliador acompanhe a mudança de contexto.
- **Troque a aba** visível no editor a cada sub-seção para manter o vídeo visualmente dinâmico.
- **Fale com confiança** na seção de Result — é o momento de mostrar orgulho do que foi construído.
- **Encerre com firmeza** — não deixe o final "morrer". A última frase deve soar como ponto final, não reticências.
