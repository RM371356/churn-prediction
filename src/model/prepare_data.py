# O arquivo prepare_data.py é responsável por carregar os dados brutos, realizar o pré-processamento e preparar os dados para o treinamento do modelo.
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_and_prepare(path):
    """Carrega os dados de um arquivo Excel, realiza o pré-processamento e retorna os dados prontos para treinamento e teste, além do preprocessor para uso futuro."""
    df = pd.read_excel(path)
    
    # Padronizar nomes das colunas (remover espaços, colocar em minúsculas)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # Definir target e colunas a serem removidas
    target_col = "churn_value"
    drop_cols = ["customerid", "count", "lat_long", "churn_label", 
                 "churn_score", "churn_reason", "city", "country", "state", "zip_code"]
    
    # Remover colunas irrelevantes
    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identificar colunas numéricas e categóricas
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns

    # Preencher valores faltantes em colunas categóricas com "missing" e garantir que sejam do tipo string
    X[cat_cols] = X[cat_cols].fillna("missing").astype("string")

    # Pipeline para colunas numéricas: preenche valores faltantes e normaliza
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline para colunas categóricas: preenche valores faltantes e aplica one-hot encoding
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combinação dos pipelines
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    
    # Configura o output do pipeline para ser um DataFrame, facilitando a conversão para tensor depois
    preprocessor.set_output(transform="pandas")

    # split antes do fit_transform para evitar vazamento de dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # transformar os dados usando o pipeline APENAS AGORA, depois de garantir que estão no formato correto
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # converter para tensor (PyTorch exige tensores numéricos)
    # O pipeline já retorna um DataFrame, então podemos converter diretamente para numpy e depois para tensor
    X_train = torch.tensor(X_train_transformed.to_numpy(), dtype=torch.float32)
    X_test = torch.tensor(X_test_transformed.to_numpy(), dtype=torch.float32)
    
    # Converter target para tensor
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # Retornar os dados preparados e o preprocessor
    return X_train, X_test, y_train, y_test, preprocessor