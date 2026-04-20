from fastapi import APIRouter
from src.app.schemas import InputData
from src.services.model_service import predict

# Define o roteador para as rotas da API
router = APIRouter()

# Rota de saúde para verificar se a API está funcionando
@router.get("/health")
def health():
    return {"status": "ok"}

# Rota para fazer previsões de churn
@router.post("/predict")
def make_prediction(data: InputData):
    return predict(data.model_dump())