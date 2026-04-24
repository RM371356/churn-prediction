import pandas as pd

def preprocess_input(data: dict, features: list, scaler: object):
    """
        Recebe um dicionário de dados de entrada, aplica o mesmo pré-processamento usado no treinamento e retorna os dados prontos para inferência.
        Args:
            data (dict): Os dados de entrada para a previsão, fornecidos como um dicionário.
            features (list): A lista de features esperadas pelo modelo, usada para garantir que o DataFrame resultante tenha as colunas corretas.
            scaler (object): O objeto de pré-processamento usado para escalar os dados, garantindo que os dados de entrada sejam processados da mesma forma que os dados de treinamento.
        Returns:
            X (np.ndarray): Os dados pré-processados prontos para inferência.
    """
    df = pd.DataFrame([data])

    # Limpar e padronizar os nomes das colunas para evitar problemas de alinhamento
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Aplicar one-hot encoding para colunas categóricas, garantindo que o resultado seja um DataFrame
    df = pd.get_dummies(df)

    # Reindexar o DataFrame para garantir que todas as colunas esperadas estejam presentes, preenchendo as ausentes com 0
    df = df.reindex(columns=features, fill_value=0)

    # Garantir que os tipos de dados estejam corretos (ex: colunas numéricas como float)
    X = scaler.transform(df)

    return X