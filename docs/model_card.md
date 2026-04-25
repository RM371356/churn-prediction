
# Model Card — Churn Prediction MLP

## Data
2026-04-24 20:46

---

## Objetivo
Este modelo foi desenvolvido com o objetivo de prever a probabilidade de churn (cancelamento) de clientes em uma empresa de telecomunicações.

A solução permite identificar clientes com maior risco de evasão, possibilitando a criação de estratégias de retenção mais eficientes.
---


## Problema de Negócio

O churn representa uma perda direta de receita para a empresa. Antecipar quais clientes possuem maior probabilidade de cancelamento permite:

Reduzir perdas financeiras
Direcionar campanhas de retenção
Melhorar a experiência do cliente

## Dados
- Dataset: Telco Customer Churn
- Número de registros: 5634
- Número de features: 5324
- Distribuição target:
{0.0: 0.7346467873624423, 1.0: 0.2653532126375577}

---

## Modelo

- Tipo: MLP (PyTorch)
- Parâmetros:
{
  "input_dim": 5324,
  "epochs": 100,
  "batch_size": 64,
  "early_stopping": true
}

---

## Métricas

- Accuracy: 0.7303
- Precision: 0.4948
- Recall: 0.7647
- F1-score: 0.6008
- ROC-AUC: 0.8118


## Pipeline
O pipeline do modelo inclui:

- Pré-processamento dos dados
- Engenharia de features
- Seleção de atributos
- Modelo de classificação

---

## Threshold

- Valor utilizado: 0.5
- Estratégia: Ajustado para maximizar recall (reduzir falso negativo)

---

## Limitações

- Pode ter viés em categorias pouco representadas
- Sensível à qualidade dos dados de entrada
- Performance depende do balanceamento de classes

---

## Uso em Produção

- API REST (FastAPI)
- Pipeline consistente com treinamento
- Preprocessor versionado
- Identificação de clientes em risco
- Apoio a campanhas de retenção
- Priorização de atendimento

---

## Atualização

Recomenda-se re-treinar o modelo periodicamente com novos dados para manter a performance e reduzir o risco de drift, fazer a atualização dos pipelines e ajustes em features.

## Versão do Modelo

 - Versão: 1.0
 - Data: 05/05/2026
 - Status: Produção
