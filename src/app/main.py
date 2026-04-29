from fastapi import FastAPI

from src.app.middleware import LatencyMiddleware
from src.app.routes import router

# Cria a aplicação FastAPI
app = FastAPI(title="Churn Prediction API")

# Adiciona o middleware de latência para medir o tempo de processamento de cada requisição e adicionar um ID único para rastreamento.
app.add_middleware(LatencyMiddleware)

# Inclui as rotas definidas no roteador do arquivo routes.py
app.include_router(router)

# Inclui as rotas definidas no roteador do arquivo routes.py
app.include_router(router)