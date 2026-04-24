from fastapi import APIRouter
from src.app.schemas import InputData
from src.services.model_service import predict
from src.utils.logger import logger

# Define o roteador para as rotas da API
router = APIRouter()

# Rota de saúde para verificar se a API está funcionando
@router.get("/health")
def health():
    """
        Endpoint de saúde para verificar se a API está funcionando. Retorna um status "ok" e registra a chamada nos logs.
        
        Returns:
            dict: Um dicionário contendo o status da API.
    """
    logger.info("Health check chamado")
    return {"status": "ok"}

# Rota para fazer previsões de churn
@router.post("/predict")
def make_prediction(data: InputData):
    """
        Recebe os dados de entrada para previsão de churn, registra a solicitação e o resultado nos logs, e retorna a previsão.
        
        Args:
            data (InputData): Os dados de entrada para a previsão, validados pelo modelo Pydantic InputData.
        
        Returns:
            dict: O resultado da previsão, contendo a probabilidade de churn e a classe prevista.
    """
    logger.info(f"Nova predição recebida: {data.model_dump()}")
    
    result = predict(data.model_dump())
    
    logger.info(f"Resultado predição: {result}")
    
    return result