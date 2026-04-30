from pydantic import BaseModel, ConfigDict, Field


class InputData(BaseModel):
    """
        Modelo de dados de entrada para a previsão de churn, utilizando Pydantic para validação e mapeamento dos campos de entrada.
        Os campos são definidos com tipos específicos e aliases para corresponder aos nomes esperados na solicitação JSON, permitindo flexibilidade na estrutura dos dados de entrada.
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    customer_id: str
    gender: str
    tenure_months: int = Field(..., alias="tenure")
    monthly_charges: float
    contract: str
    internet_service: str
    payment_method: str