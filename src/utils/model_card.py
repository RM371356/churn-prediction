from datetime import datetime
import json
from pathlib import Path


def generate_model_card(
    model_name: str,
    metrics: dict,
    threshold: float,
    dataset_info: dict,
    model_params: dict,
    output_path: str = "model_card.md"
):
    """Gera um model card automático em Markdown com base nas métricas de avaliação, informações do dataset e parâmetros do modelo."""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    content = f"""
# Model Card — {model_name}

## Data
{now}

---

## Objetivo
Modelo para previsão de churn de clientes.

---

## Dados

- Número de registros: {dataset_info.get("n_samples")}
- Número de features: {dataset_info.get("n_features")}
- Distribuição target:
{dataset_info.get("target_distribution")}

---

## Modelo

- Tipo: MLP (PyTorch)
- Parâmetros:
{json.dumps(model_params, indent=2)}

---

## Métricas

- Accuracy: {metrics.get("accuracy"):.4f}
- Precision: {metrics.get("precision"):.4f}
- Recall: {metrics.get("recall"):.4f}
- F1-score: {metrics.get("f1"):.4f}
- ROC-AUC: {metrics.get("roc_auc"):.4f}

---

## Threshold

- Valor utilizado: {threshold}
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
"""

    Path(output_path).write_text(content, encoding="utf-8")

    print(f"Model card gerado em: {output_path}")