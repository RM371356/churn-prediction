import numpy as np


def preprocess(data: dict):
    """
        Transformar o JSON de entrada em um tensor de modelo, aplicando as mesmas transformações usadas no treinamento.
        Args:
            data (dict): Dicionário contendo os dados de entrada para a previsão, onde as chaves são os nomes das features e os valores são os valores correspondentes.
        Returns:
           features (np.ndarray): Tensor do modelo.
    """
    features = np.array(list(data.values()), dtype=float)
    return features
