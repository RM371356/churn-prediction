import logging
import sys


def setup_logger():
    """
        Configura um registrador de dados para a API de churn.
        Returns:
            logging.Logger: Instância de registro configurada.
    """
    # Verifica se o logger já possui manipuladores para evitar a adição de múltiplos manipuladores em caso de múltiplas chamadas a `setup_logger`.
    logger = logging.getLogger("churn_api")

    if logger.handlers:
        return logger

    # Define o nível de log para INFO, o que significa que mensagens de nível INFO e superiores (WARNING, ERROR, CRITICAL) serão registradas.
    logger.setLevel(logging.INFO)

    # Define um formato de log que inclui a data e hora, o nível de log, o nome do logger e a mensagem de log. O formato da data é definido como "YYYY-MM-DD HH:MM:SS".
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Cria um manipulador de fluxo que envia a saída para stdout e define o formato para ele.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Adiciona o manipulador ao logger e define propagate como False para evitar que as mensagens de log sejam propagadas para o logger raiz.
    logger.addHandler(handler)
    logger.propagate = False

    return logger


logger = setup_logger()