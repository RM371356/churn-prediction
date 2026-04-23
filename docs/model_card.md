
# Model Card — Churn Prediction MLP

## Data
2026-04-23 19:18

---

## Objetivo
Modelo para previsão de churn de clientes.

---

## Dados

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

---

## Atualização

Recomenda-se re-treinar o modelo periodicamente com novos dados para manter a performance e reduzir o risco de drift.
