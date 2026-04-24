import torch
import joblib
import pandas as pd
from typing import Dict, Any

from src.config.settings import MODEL_PATH, PREPROCESSOR_PATH, THRESHOLD
from src.model.mlp import MLP

# Inicializa recursos globais
_MODEL = None
_PREPROCESSOR = None

def get_resources():
    """
        Carrega artefatos sob demanda (lazy load) e armazena em cache no escopo global.
        Retorna o modelo e o pré-processador carregados, garantindo que sejam carregados apenas uma vez durante a vida útil da aplicação.
        Returns:
            tuple: Uma tupla contendo o modelo e o pré-processador carregados.
    """
    # Usar variáveis globais para armazenar o modelo e o pré-processador carregados
    global _MODEL, _PREPROCESSOR
    if _MODEL is None:
        
        # Carregar modelo e pré-processador do disco apenas na primeira chamada
        model_data = torch.load(MODEL_PATH)
        _PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
        
        # Criar instância do modelo e carregar os pesos
        _MODEL = MLP(model_data["input_dim"])
        _MODEL.load_state_dict(model_data["model_state"])
        _MODEL.eval()
        
    return _MODEL, _PREPROCESSOR

def predict(data: Dict[str, Any]) -> Dict[str, float]:
    """
        Realiza a previsão de churn para um cliente com base nos dados de entrada fornecidos.
        Args:
            data (Dict[str, Any]): Um dicionário contendo os dados de entrada para a previsão, onde as chaves são os nomes das features e os valores são os valores correspondentes.
        Returns:
            Dict[str, float]: Um dicionário contendo a probabilidade de churn e a previsão binária.
    """
    # Carregar recursos (modelo e pré-processador)
    model, preprocessor = get_resources()

    # Criar DataFrame a partir do dicionário de entrada
    df = pd.DataFrame([data])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Alinhar colunas com o que o pré-processador espera (adicionar colunas ausentes como None)
    expected_cols = preprocessor.feature_names_in_
    
    # Garantir que todas as colunas esperadas estejam presentes, preenchendo as ausentes com None
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    
    # Reordenar colunas para garantir que estejam na ordem esperada pelo modelo
    df = df[list(expected_cols)]

    # Converter tipos de dados para evitar erros no pipeline (ex: colunas numéricas como string)
    df = df.convert_dtypes() 

    # Transformar os dados usando o pré-processador
    with torch.no_grad():
        # O pré-processador pode retornar um DataFrame ou um array, dependendo da configuração. Garantir que seja um array para o modelo.
        X = preprocessor.transform(df)
        
        # Converter para numpy array se for um DataFrame, garantindo compatibilidade com o modelo PyTorch
        X = X.to_numpy() if hasattr(X, "to_numpy") else X
        
        # Converter para tensor do PyTorch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Fazer a previsão usando o modelo carregado
        logits = model(X_tensor)

        # Aplicar sigmoid para obter a probabilidade de churn
        prob = torch.sigmoid(logits).item()

    return {
        "probability": prob,
        "prediction": int(prob > THRESHOLD)
    }