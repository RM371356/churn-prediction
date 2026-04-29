# O arquivo predict.py é responsável por carregar o modelo treinado e realizar previsões com novos dados de entrada.
# Define funções para carregar o modelo e fazer previsões, retornando as probabilidades de churn para cada cliente.
import torch
from mlp import MLP


def load_model(input_dim):
    """
        Carrega o modelo treinado a partir do arquivo salvo.
        Args:
            input_dim: O número de features de entrada para o modelo, necessário para criar a arquitetura correta da rede.
        Returns:
            O modelo carregado pronto para fazer previsões.
    """

    # Criar uma nova instância do modelo com o input_dim correto
    model = MLP(input_dim)

    # Carregar os pesos do modelo salvo
    model.load_state_dict(torch.load("saved_models/model.pt"))
    
    # Colocar o modelo em modo de avaliação
    model.eval()
    
    return model


def predict(model, X):
    """
        Faz previsões usando o modelo carregado e retorna as probabilidades de churn.
        Args:
            model: O modelo carregado.
            X: Os dados de entrada (features) para os quais fazer previsões.
        Returns:
            As probabilidades de churn para cada cliente.
    """
    with torch.no_grad():
        # Fazer a previsão usando o modelo carregado
        logits = model(X)
        
        # Aplicar sigmoid para obter a probabilidade de churn
        probs = torch.sigmoid(logits)
        return probs.numpy()