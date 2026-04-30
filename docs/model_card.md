
# Model Card — Churn Prediction MLP

## Data
2026-04-29 20:52

---

## Objetivo
Modelo para previsão de churn de clientes.
O objetivo é identificar clientes com alta probabilidade de cancelamento para ações de retenção.

---

## Dados

- Número de registros: 5634
- Número de features: 5321
- Distribuição do target:
{
  "churn": 1495,
  "no_churn": 4139
}
---

## Modelo

- Tipo: MLP (PyTorch)
- Parâmetros:
{
  "input_dim": 5321,
  "epochs": 6.4034013950731605,
  "batch_size": 64,
  "early_stopping": true
}
- Pré-processamento: Imputação de valores ausentes e escalonamento (StandardScaler)
- Treinado com BCEWithLogitsLoss e otimizador Adam
- Peso para classe positiva ajustado para lidar com desbalanceamento
- Treinado por 6.4034013950731605 épocas com batch size de 64
- Early stopping aplicado para evitar overfitting
- Threshold de classificação ajustado para maximizar recall, reduzindo falsos negativos
- Peso para classe positiva: 2.77
- Modelo avaliado usando métricas de acurácia, precisão, recall, F1-score e AUC-ROC
- Modelo salvo para uso em produção via API REST (FastAPI)

---

## Métricas

- Accuracy: 0.7637
- Precision: 0.5411
- Recall: 0.7219
- F1-score: 0.6186
- ROC-AUC: 0.8270

---

## Threshold

- Valor utilizado: 0.30000000000000004
- Estratégia: Ajustado para maximizar recall (reduzir falso negativo), importante para retenção de clientes.
- Sensível ao balanceamento entre precisão e recall, escolha do threshold impacta diretamente na performance do modelo em produção.
- Recomendação: Monitorar e ajustar o threshold periodicamente com base no feedback do modelo em produção para manter a performance ideal.

---

## Limitações

- Pode ter viés em categorias pouco representadas
- Sensível à qualidade dos dados de entrada
- Performance depende do balanceamento de classes
- Pode não capturar relações complexas entre features sem ajustes adicionais
- Modelo pode ser afetado por drift de dados ao longo do tempo, exigindo monitoramento contínuo
- Modelo pode não generalizar bem para dados muito diferentes do conjunto de treinamento, exigindo validação cuidadosa antes de uso em produção
- Modelo pode ser sensível a outliers, exigindo tratamento adequado dos dados de entrada para garantir previsões confiáveis

---

## Uso em Produção

- API REST (FastAPI)
- Pipeline consistente com treinamento
- Preprocessor versionado para garantir consistência
- Monitoramento de performance e drift de dados recomendado
- Logs de previsões e métricas para análise contínua

---

## Atualização

- Recomenda-se re-treinar o modelo periodicamente com novos dados para manter a performance e reduzir o risco de drift.
- Monitorar métricas de performance em produção e ajustar o modelo ou o threshold conforme necessário para garantir que o modelo continue a atender aos objetivos de negócio.
- Manter o model card atualizado com as informações mais recentes sobre o modelo, métricas e uso em produção para garantir transparência e facilitar a comunicação com stakeholders.
- Documentar quaisquer mudanças significativas no modelo, como alterações na arquitetura, parâmetros ou dados de treinamento, para manter um histórico claro do desenvolvimento do modelo ao longo do tempo.

