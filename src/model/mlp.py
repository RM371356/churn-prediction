# O arquivo mlp.py define a arquitetura da rede neural MLP (Multi-Layer Perceptron) para o modelo de previsão de churn.

import torch.nn as nn

# A classe MLP herda de nn.Module, que é a base para todos os modelos em PyTorch
class MLP(nn.Module):
    """Definição da arquitetura da rede neural MLP para previsão de churn."""
    def __init__(self, input_dim):
        super().__init__()

        # Arquitetura da rede
        self.net = nn.Sequential(
            # Camada de entrada com mais neurônios para capturar complexidade
            nn.Linear(input_dim, 128),
            
            # Camada oculta com ReLU, normalização e dropout para evitar overfitting
            nn.ReLU(),

            # Adicionar normalização em lote para melhorar a estabilidade do treinamento
            nn.BatchNorm1d(128),

            # Aumentar o dropout para reduzir ainda mais o risco de overfitting, especialmente com uma arquitetura mais complexa
            nn.Dropout(0.4),

            # Camada oculta adicional para permitir que o modelo capture interações mais complexas entre as features
            nn.Linear(128, 64),

            # ReLU e dropout para a segunda camada oculta
            nn.ReLU(),

            # Normalização em lote para a segunda camada oculta
            nn.BatchNorm1d(64),

            # Dropout para a segunda camada oculta
            nn.Dropout(0.3),

            # Camada de saída com um neurônio para previsão binária (churn ou não churn)
            nn.Linear(64, 32),

            # ReLU e dropout para a terceira camada oculta
            nn.ReLU(),

            # Normalização em lote para a terceira camada oculta
            nn.Dropout(0.2),

            # Camada de saída final para previsão binária
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)