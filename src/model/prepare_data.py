import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.monitoring.drift_monitor import save_baseline


def load_and_prepare(path):
    """
        Carrega os dados de um arquivo Excel, realiza o pré-processamento necessário, e retorna os conjuntos de treino e teste para as features e o target, 
        juntamente com o pré-processador utilizado. A função também salva as estatísticas de baseline para as features numéricas, 
        que serão usadas posteriormente para monitorar o drift dos dados. O pré-processamento inclui a imputação de valores ausentes, 
        o escalonamento das features numéricas e a codificação das features categóricas usando OneHotEncoder, 
        garantindo que os dados estejam prontos para o treinamento do modelo de previsão de churn, 
        e que as estatísticas de baseline estejam disponíveis para a detecção de drift no futuro.
        Args:
            path (str): O caminho para o arquivo Excel contendo os dados a serem carregados e preparados.
        Returns:
            tuple: Uma tupla contendo os conjuntos de treino e teste para as features e o target, juntamente com o pré-processador utilizado.
    """
    df = pd.read_excel(path)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    target_col = "churn_value"

    # remover colunas irrelevantes ou que possam causar vazamento de dados, garantindo que o modelo seja treinado apenas com as features relevantes para a previsão de churn
    drop_cols = [
        "customerid", "count", "country", "state", "city",
        "zip_code", "lat_long", "latitude", "longitude",
        "churn_label", "churn_score", "cltv", "churn_reason",
    ]

    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # salvar as estatísticas de baseline para as features numéricas, que serão usadas posteriormente para monitorar o drift dos dados
    save_baseline(X)

    # identificar colunas numéricas e categóricas para aplicar os pipelines de pré-processamento adequados
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns

    # garantir que as colunas categóricas sejam tratadas como strings e que os valores ausentes sejam preenchidos com uma categoria "missing" para evitar problemas durante a codificação
    X[cat_cols] = X[cat_cols].fillna("missing").astype("string")

    # definir pipelines de pré-processamento para colunas numéricas e categóricas, garantindo que os dados sejam adequadamente transformados para o treinamento do modelo
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # o pipeline para colunas categóricas inclui a imputação de valores ausentes com a categoria "missing" e a codificação usando OneHotEncoder
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # combinar os pipelines de pré-processamento para colunas numéricas e categóricas usando ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # aplicar o pré-processamento aos dados, garantindo que as features sejam transformadas de acordo com os pipelines definidos
    preprocessor.set_output(transform="pandas")

    # dividir os dados em conjuntos de treino e teste, garantindo que a divisão seja estratificada com base no target para manter a proporção de classes em ambos os conjuntos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # aplicar o pré-processamento aos conjuntos de treino e teste, garantindo que as transformações 
    # sejam aplicadas de forma consistente e que os dados estejam prontos para o treinamento do modelo
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return (
        torch.tensor(X_train.to_numpy(), dtype=torch.float32),
        torch.tensor(X_test.to_numpy(), dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32),
        preprocessor,
    )