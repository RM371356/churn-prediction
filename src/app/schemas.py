from pydantic import BaseModel, ConfigDict, Field


class InputData(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    gender: str
    tenure_months: int = Field(..., alias="tenure")
    monthly_charges: float
    contract: str
    internet_service: str
    payment_method: str