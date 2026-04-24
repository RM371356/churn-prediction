import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

from src.utils.logger import logger


class LatencyMiddleware(BaseHTTPMiddleware):
    """
        Middleware para medir a latência de cada requisição e adicionar um ID único para rastreamento.
    """
    async def dispatch(self, request: Request, call_next):
        """Processa a requisição, mede o tempo de processamento e adiciona informações de latência e ID de requisição nos logs e cabeçalhos da resposta.
        Args:
            request (Request): A requisição HTTP recebida.
            call_next: Função para chamar o próximo middleware ou endpoint.
            Returns:
                Response: A resposta HTTP processada com os cabeçalhos de latência e ID de requisição.
        """
        # Registra o tempo de início da requisição para calcular a latência posteriormente.
        start_time = time.time()

        # request id único para rastreamento de logs relacionados a esta requisição específica.
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Loga o início da requisição, incluindo o ID da requisição, método HTTP e caminho da URL.
        logger.info(
            f"Request started | id={request_id} "
            f"| method={request.method} "
            f"| path={request.url.path}"
        )

        # Chama o próximo middleware ou endpoint para processar a requisição e obter a resposta.
        response = await call_next(request)

        # Calcula o tempo de processamento da requisição em milissegundos e arredonda para 2 casas decimais.
        process_time = round((time.time() - start_time) * 1000, 2)

        # Adiciona os cabeçalhos "X-Process-Time-ms" e "X-Request-ID" na resposta para fornecer informações sobre a latência e o ID da requisição.
        response.headers["X-Process-Time-ms"] = str(process_time)
        response.headers["X-Request-ID"] = request_id

        # Loga o término da requisição, incluindo o ID da requisição, status da resposta e latência em milissegundos.
        logger.info(
            f"Request finished | id={request_id} "
            f"| status={response.status_code} "
            f"| latency_ms={process_time}"
        )

        return response