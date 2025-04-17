import logging
from rich.logging import RichHandler
import traceback

# Настройки форматирования
FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="%H:%M:%S",
    handlers=[RichHandler(markup=True)]
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def full_log(logger: logging.Logger, where: str) -> None:
    tb = traceback.format_exc()
    logger.info(f"=== Exception in {where} ===")
    logger.info(tb)
