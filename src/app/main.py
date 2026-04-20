from fastapi import FastAPI
from src.app.routes import router

# Cria a aplicação FastAPI
app = FastAPI(title="Churn Prediction API")

# Inclui as rotas definidas no roteador do arquivo routes.py
app.include_router(router)