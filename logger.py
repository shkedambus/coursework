import logging

from rich.logging import RichHandler

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
