import json
from datetime import datetime
from pathlib import Path


def generate_model_card(
    model_name: str,
    metrics: dict,
    threshold: float,
    dataset_info: dict,
    model_params: dict,
    output_path: str = "model_card.md"
):
    """
        Gera um model card automático em Markdown com base nas métricas de avaliação, informações do dataset e parâmetros do modelo.
        Args:
            model_name (str): O nome do modelo para incluir no model card.
            metrics (dict): Um dicionário contendo as métricas de avaliação do modelo (accuracy, precision, recall, f1, roc_auc).
            threshold (float): O valor do limiar utilizado para classificar as previsões, importante para incluir no model card para transparência.
            dataset_info (dict): Um dicionário contendo informações sobre o dataset utilizado para treinamento, como número de amostras, número de features e distribuição do target.
            model_params (dict): Um dicionário contendo os parâmetros do modelo, como número de features de entrada, número de épocas, batch size, etc., para incluir no model card e fornecer contexto sobre a configuração do modelo.
            output_path (str): O caminho onde o model card em Markdown será salvo. O padrão é "model_card.md".
        Returns:
            None: A função salva o model card em um arquivo Markdown no caminho especificado.
    """

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    content = f"""
# Model Card — {model_name}

## Data
{now}

---

## Objetivo
Modelo para previsão de churn de clientes.
O objetivo é identificar clientes com alta probabilidade de cancelamento para ações de retenção.

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
- Pré-processamento: Imputação de valores ausentes e escalonamento (StandardScaler)
- Treinado com BCEWithLogitsLoss e otimizador Adam
- Peso para classe positiva ajustado para lidar com desbalanceamento
- Treinado por {model_params.get("epochs")} épocas com batch size de {model_params.get("batch_size")}
- Early stopping aplicado para evitar overfitting
- Threshold de classificação ajustado para maximizar recall, reduzindo falsos negativos
- Modelo avaliado usando métricas de acurácia, precisão, recall, F1-score e AUC-ROC
- Modelo salvo para uso em produção via API REST (FastAPI)

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

"""

    Path(output_path).write_text(content, encoding="utf-8")

    print(f"Model card gerado em: {output_path}")