# Pipeline completo de treinamento do modelo de churn prediction
import os
import pandas as pd
import torch
import logging
import joblib
from src.model.mlp import MLP
from src.model.prepare_data import load_and_prepare
from src.model.evaluate import evaluate

from src.utils.model_card import generate_model_card
from src.config.settings import DATA_PATH, MODEL_DIR, MODEL_PATH, PREPROCESSOR_PATH, EPOCH, MODEL_CARD_PATH, THRESHOLD

# Garantir que o diretório para salvar o modelo exista
os.makedirs(MODEL_DIR, exist_ok=True)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training():
    """Função principal para executar o pipeline de treinamento do modelo de churn prediction."""
    logger.info("Iniciando pipeline de treino")

    # Carrega e prepara os dados
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare(
        DATA_PATH
    )
    
    # Converter os dados para tensores do PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Calcular peso para classe positiva (churn) para lidar com desbalanceamento
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Peso para classe positiva: {pos_weight.item():.2f}")

    # Determinar o número de features de entrada para o modelo
    input_dim = X_train.shape[1]

    # Criar instância do modelo
    model = MLP(input_dim)

    # Configurar critério de perda e otimizador
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Treinar o modelo por um número fixo de épocas
    for epoch in range(EPOCH):
        
        # Colocar o modelo em modo de treinamento
        optimizer.zero_grad()

        # Fazer previsões e calcular a perda
        preds = model(X_train).squeeze()
        loss = criterion(preds, y_train)

        # Backpropagation e atualização dos pesos
        loss.backward()
        optimizer.step()

        # Logar a perda a cada 10 épocas para monitorar o progresso do treinamento
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch} | Loss {loss.item():.4f}")

    # Avaliar o modelo usando os dados de teste
    metrics = evaluate(model, X_test, y_test)
    logger.info(f"Métricas finais: {metrics}")

    # Informações do dataset para incluir no model card
    dataset_info = {
        "n_samples": len(y_train),
        "n_features": X_train.shape[1],
        "target_distribution": pd.Series(y_train).value_counts(normalize=True).to_dict()
    }

    # Parâmetros do modelo para incluir no model card
    model_params = {
        "input_dim": X_train.shape[1],
        "epochs": EPOCH,
        "batch_size": 64,
        "early_stopping": True
    }

    # Gerar o model card com as informações do modelo, métricas e dataset
    generate_model_card(
        model_name="Churn Prediction MLP",
        metrics=metrics,
        threshold=THRESHOLD,
        dataset_info=dataset_info,
        model_params=model_params,
        output_path=MODEL_CARD_PATH
    )

    # Garantir que o diretório para salvar o modelo exista
    os.makedirs("saved_models", exist_ok=True)

    # Salvar o modelo treinado e o pré-processador para uso futuro na API
    torch.save({
        "model_state": model.state_dict(),
        "input_dim": input_dim
    }, MODEL_PATH)

    # Garantir que o diretório para salvar o pré-processador exista
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Salvar o pré-processador usando joblib para que possa ser carregado posteriormente na API
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

# Executar o pipeline de treinamento quando este script for executado diretamente
if __name__ == "__main__":
    run_training()