import joblib
import pandas as pd
import torch

from src.config.settings import MODEL_PATH, PREPROCESSOR_PATH
from src.model.mlp import MLP

_MODEL = None
_PREPROCESSOR = None
_THRESHOLD = 0.5


def load_resources():
    """
        Carrega o modelo, o pré-processador e o threshold do disco, 
        garantindo que os recursos necessários para a inferência estejam disponíveis na memória, 
        e evitando recarregamentos desnecessários em chamadas subsequentes à função de previsão, 
        otimizando o desempenho e a eficiência do serviço de modelo ao manter os recursos carregados em memória após a primeira carga, 
        e garantindo que o modelo e o pré-processador sejam carregados apenas uma vez durante a vida útil do serviço, 
        melhorando a eficiência e a velocidade das previsões subsequentes. 
        Returns:
            tuple: Uma tupla contendo o modelo carregado, o pré-processador e o threshold para classificação.
    """
    global _MODEL, _PREPROCESSOR, _THRESHOLD

    if _MODEL is None:
        data = torch.load(MODEL_PATH)

        _MODEL = MLP(data["input_dim"])
        _MODEL.load_state_dict(data["model_state"])
        _MODEL.eval()

        _PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
        _THRESHOLD = data.get("threshold", 0.5)

    return _MODEL, _PREPROCESSOR, _THRESHOLD


def predict(data: dict):
    """        
        Realiza a previsão de churn com base nos dados de entrada fornecidos, utilizando o modelo e o pré-processador carregados,
        e aplicando o threshold para determinar a classe de churn ou não churn, 
        garantindo que os dados de entrada sejam pré-processados corretamente antes de serem alimentados no modelo, 
        e que a saída da previsão seja interpretada de acordo com o threshold definido para fornecer uma resposta clara sobre a probabilidade de churn e a classificação resultante.
        Args:
            data (dict): Um dicionário contendo os dados de entrada para a previsão, onde as chaves correspondem aos nomes das features esperadas pelo modelo, 
            e os valores são os dados específicos para a previsão.
        Returns:
            dict: Um dicionário contendo a probabilidade prevista de churn e a classificação binária (1 para churn, 0 para não churn) com base no threshold definido.
    """
    model, preprocessor, threshold = load_resources()

    df = pd.DataFrame([data])
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    expected_cols = preprocessor.feature_names_in_

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df = df[expected_cols]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
        if df[col].dtype == "object":
            df[col] = df[col].astype("string")

    X = preprocessor.transform(df)
    X = X.to_numpy() if hasattr(X, "to_numpy") else X

    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(X)).item()

    return {
        "probability": prob,
        "prediction": int(prob > threshold)
    }