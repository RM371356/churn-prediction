from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.utils.logger import logger

PREDICTIONS_LOG_PATH = Path("monitoring/predictions_log.csv")


def log_prediction(customer_id, prediction, probability):
    """Registra a previsão de churn para um cliente específico, incluindo a probabilidade prevista e o rótulo real (se disponível).
    Args:        customer_id (str): O identificador único do cliente.
        prediction (int): A previsão binária de churn (0 ou 1).
        probability (float): A probabilidade prevista de churn (entre 0 e 1).
        actual (int, opcional): O rótulo real de churn (0 ou 1), se disponível. Padrão é None.
    """
    # Criar um DataFrame para a nova linha de previsão
    row = pd.DataFrame(
        [
            {
                "customer_id": customer_id,
                "prediction": prediction,
                "probability": probability
            }
        ]
    )

    # Garantir que o diretório exista antes de tentar salvar o arquivo
    PREDICTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Salvar a previsão no arquivo CSV, adicionando uma nova linha ou criando o arquivo se ele não existir
    if PREDICTIONS_LOG_PATH.exists():
        row.to_csv(PREDICTIONS_LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(PREDICTIONS_LOG_PATH, index=False)


def evaluate_real_business_metrics():
    """
        Calcula métricas de negócio reais (precisão, recall, f1) com base nas previsões registradas e nos rótulos reais disponíveis.
        Retorna:
            dict: Um dicionário contendo as métricas calculadas (precisão, recall, f1) ou None se as métricas não puderem ser calculadas devido à falta de dados.
    """
    # Verificar se o arquivo de log de previsões existe antes de tentar ler os dados
    if not PREDICTIONS_LOG_PATH.exists():
        logger.warning("prediction_log_not_found")
        return

    # Carregar os dados de previsões registradas
    df = pd.read_csv(PREDICTIONS_LOG_PATH)

    # Filtrar apenas as linhas onde o rótulo real (actual) está disponível para calcular as métricas de negócio
    df = df.dropna(subset=["actual"])

    # Se não houver rótulos reais disponíveis, não é possível calcular as métricas de negócio, então registramos um aviso e retornamos
    if df.empty:
        logger.warning("no_actual_labels_available")
        return

    # Calcular as métricas de negócio usando os rótulos reais e as previsões registradas
    y_true = df["actual"]
    y_pred = df["prediction"]

    # Calcular precisão, recall e f1 score, arredondando os resultados para 4 casas decimais
    metrics = {
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }
    
    logger.info(f"business_metrics={metrics}")

    return metrics