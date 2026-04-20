import torch
from src.model.mlp import MLP

# Carrega o modelo salvo e prepara para inferência
INPUT_DIM = 4

# Criar instância do modelo e carregar os pesos
model = MLP(INPUT_DIM)

# Carregar os pesos do modelo salvo
saved = torch.load("saved_models/model.pt", map_location="cpu")

# Atualizar o modelo com os pesos carregados
INPUT_DIM = saved['input_dim']

# Criar nova instância do modelo com o input_dim correto
model = MLP(INPUT_DIM)

# Carregar os pesos do modelo salvo
model.load_state_dict(saved['model_state'])

# Colocar o modelo em modo de avaliação
model.eval()

# Função para obter o modelo carregado
def get_model():
    return model
