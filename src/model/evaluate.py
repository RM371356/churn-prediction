# Responsável por calcular métricas de avaliação do modelo, como acurácia, precisão, recall, F1-score e AUC-ROC.
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X_test, y_test):
    """
        Avalia o modelo usando os dados de teste e retorna um dicionário com as métricas de avaliação.
        Args:
            model: O modelo treinado a ser avaliado.
            X_test: Os dados de teste (features).
            y_test: Os rótulos verdadeiros para os dados de teste.
        Returns:
            dict: Um dicionário contendo as métricas de avaliação (accuracy, precision, recall, f1, roc_auc).
    """
    model.eval()

    # Converter os dados de teste para tensor
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # Fazer previsões e calcular probabilidades
    with torch.no_grad():
        logits = model(X_test).squeeze()
        probs = torch.sigmoid(logits).numpy()

    # Converter probabilidades em classes binárias usando um limiar de 0.5
    preds = (probs > 0.5).astype(int)

    # Calcular métricas de avaliação
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    
    return metrics
