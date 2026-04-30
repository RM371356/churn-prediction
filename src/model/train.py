import logging

import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.settings import (
    DATA_PATH,
    MODEL_CARD_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PREPROCESSOR_PATH,
)
from src.model.evaluate import evaluate
from src.model.mlp import MLP
from src.model.prepare_data import load_and_prepare
from src.model.threshold_tuning import find_best_threshold
from src.utils.model_card import generate_model_card

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training():
    logger.info("Iniciando treino")

    # carregar e preparar os dados, incluindo pré-processamento e divisão em treino/teste, 
    # garantindo que os dados estejam prontos para o treinamento do modelo, 
    # com as features adequadamente transformadas e os rótulos de churn separados para avaliação posterior do modelo
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare(DATA_PATH)

    # Dataset + batching
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # class weight
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"pos_weight={pos_weight.item():.2f}")

    # modelo MLP simples, sem camadas ocultas, para evitar overfitting dado o tamanho do dataset e a natureza do problema, 
    # focando em uma arquitetura mais simples que pode generalizar melhor, 
    # especialmente considerando o desbalanceamento das classes e a necessidade de maximizar o recall para a classe de churn 
    # (identificar o máximo possível de casos de churn, mesmo que isso gere mais falsos positivos)
    model = MLP(X_train.shape[1])

    # definir a função de perda com peso para classe positiva para lidar com o desbalanceamento
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # inicializar a melhor perda com um valor infinito para garantir que qualquer perda calculada durante o treinamento seja considerada uma melhoria, 
    # permitindo que o modelo salve a melhor versão de si mesmo com base na perda mais baixa alcançada durante o processo de treinamento, 
    # e aplicando early stopping se a perda não melhorar por um número definido de épocas (patience), 
    # evitando overfitting e garantindo que o modelo generalize melhor para dados não vistos durante o treinamento
    best_loss = float("inf") 
    patience = 10
    counter = 0

    for epoch in range(100):
        epoch_loss = 0

        for xb, yb in loader:
            # zerar os gradientes antes de cada passo de otimização para evitar acúmulo de gradientes de épocas anteriores, 
            # garantindo que o modelo aprenda apenas com os dados do batch atual
            optimizer.zero_grad()

            # calcular as previsões do modelo para o batch atual, 
            # calcular a perda usando a função de perda definida (BCEWithLogitsLoss) e realizar o backpropagation para atualizar os pesos do modelo com base na perda calculada, 
            # acumulando a perda total para monitoramento do treinamento
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)

            # realizar o backpropagation para calcular os gradientes dos pesos do modelo com base na perda calculada e atualizar os pesos do modelo usando o otimizador Adam, 
            # garantindo que o modelo aprenda a partir dos dados do batch atual e melhore suas previsões ao longo do tempo
            loss.backward()
            optimizer.step()

            # acumular a perda total para o epoch atual, permitindo monitorar o progresso do treinamento e aplicar early stopping se necessário, 
            # garantindo que o modelo não continue treinando por muitas épocas sem melhoria na perda, o que pode levar ao overfitting
            epoch_loss += loss.item()

        logger.info(f"Epoch {epoch} | Loss {epoch_loss:.4f}")

        # verificar se a perda melhorou para aplicar early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1

        # aplicar early stopping para evitar overfitting, parando o treinamento se a perda não melhorar por um número definido de épocas (patience)
        if counter >= patience:
            logger.info("Early stopping")
            break

    # avaliação no conjunto de teste
    model.eval()
    with torch.no_grad():
        logits = model(X_test).squeeze()
        probs = torch.sigmoid(logits).numpy()

    # calcular as métricas de avaliação do modelo no conjunto de teste usando a função evaluate, 
    # fornecendo uma visão abrangente do desempenho do modelo, incluindo métricas como accuracy, 
    # precision, recall, f1-score e AUC-ROC, que são essenciais para entender a eficácia do modelo na previsão de churn, 
    # especialmente considerando o desbalanceamento das classes e a importância de maximizar o recall para a classe de churn
    metrics = evaluate(model, X_test, y_test)
    logger.info(f"Métricas: {metrics}")

    # threshold tuning para maximizar recall (minimizar falsos negativos, que são os casos de churn não identificados)
    best_threshold = find_best_threshold(y_test.numpy(), probs)
    logger.info(f"Best threshold: {best_threshold:.2f}")

    # salvar modelo e pré-processador para produção
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Salvar o modelo treinado e o pré-processador para uso futuro em produção, 
    # garantindo que ambos sejam armazenados de forma segura e acessível para a API REST que fará as previsões em tempo real
    torch.save({
        "model_state": model.state_dict(),
        "input_dim": X_train.shape[1],
        "threshold": best_threshold
    }, MODEL_PATH)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # Informações do dataset para incluir no model card
    dataset_info = {
        "n_samples": len(y_train),
        "n_features": X_train.shape[1],
        "target_distribution": {
            "churn": int((y_train == 1).sum()),
            "no_churn": int((y_train == 0).sum())
        }
    }

    # Parâmetros do modelo para incluir no model card
    model_params = {
        "input_dim": X_train.shape[1],
        "epochs": epoch_loss,
        "batch_size": 64,
        "early_stopping": True
    }

    # Gerar o model card com as informações do modelo, métricas e dataset
    generate_model_card(
        model_name="Churn Prediction MLP",
        metrics=metrics,
        threshold=best_threshold,
        dataset_info=dataset_info,
        model_params=model_params,
        output_path=MODEL_CARD_PATH,
        pos_weight=pos_weight.item()
    )


if __name__ == "__main__":
    run_training()