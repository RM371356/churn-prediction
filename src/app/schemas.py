from pydantic import BaseModel, Field

# Define o modelo de dados de entrada para a API usando Pydantic
class InputData(BaseModel):
    gender: str
    tenure_months: int = Field(..., alias="tenure")
    monthly_charges: float
    contract: str
    internet_service: str
    payment_method: str

    # Configurações adicionais para o modelo Pydantic
    class Config:
        # Permitir a população de campos usando os nomes dos campos (alias) e ignorar campos extras
        populate_by_name = True
        extra = "ignore"