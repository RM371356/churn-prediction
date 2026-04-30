import json
from pathlib import Path

import pandas as pd

from src.utils.logger import logger

BASELINE_PATH = Path("src/saved_models/baseline_stats.json")

EXCLUDED_COLUMNS = [
    "customerid",
    "customer_id",
    "churn_value",
    "churn_label",
    "churn_score",
    "churn_reason",
    "cltv",
    "latitude",
    "longitude",
    "zip_code",
    "count",
    "lat_long",
]

def save_baseline(df: pd.DataFrame):
    """
        Calcula e salva as estatísticas de baseline (média e desvio padrão) para as colunas numéricas do DataFrame fornecido.
        Args:        
            df (pd.DataFrame): O DataFrame contendo os dados para calcular as estatísticas de baseline.
    """
    stats = {}

    # remove colunas indesejadas
    df = df.drop(columns=EXCLUDED_COLUMNS, errors="ignore")

    # Calcular a média e o desvio padrão para cada coluna numérica do DataFrame e armazenar os resultados em um dicionário
    for col in df.select_dtypes(include=["number"]).columns:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        }

    # Garantir que o diretório exista antes de tentar salvar o arquivo de baseline e salvar as estatísticas calculadas em um arquivo JSON para uso futuro na detecção de drift
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info("baseline_saved")


def check_drift(df: pd.DataFrame, threshold: float = 0.30):
    """
        Verifica a existência de drift nos dados atuais em comparação com as estatísticas de baseline salvas, usando uma variação percentual como critério de detecção.
        Args:
            df (pd.DataFrame): O DataFrame contendo os dados atuais para verificar o drift.
            threshold (float, opcional): O limiar de variação percentual para considerar que houve drift. Padrão é 0.30 (30%).
    """
    
    # Verificar se o arquivo de baseline existe antes de tentar ler as estatísticas de baseline para comparação com os dados atuais
    if not BASELINE_PATH.exists():
        logger.warning("baseline_not_found")
        return

    # Carregar as estatísticas de baseline salvas e comparar com as estatísticas atuais dos dados fornecidos, calculando a variação percentual e registrando um aviso se a variação exceder o limiar definido, indicando a detecção de drift para a coluna específica
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    for col, values in baseline.items():
        if col not in df.columns:
            continue

        current_mean = float(df[col].mean())
        base_mean = values["mean"]

        if base_mean == 0:
            continue

        variation = abs((current_mean - base_mean) / base_mean)

        if variation > threshold:
            logger.warning(
                f"drift_detected column={col} baseline_mean={base_mean} current_mean={current_mean} variation={round(variation, 4)}"
            )