import logging
from rich.logging import RichHandler
import traceback
import csv
from pathlib import Path
from datetime import datetime

# Настройки форматирования
FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="%H:%M:%S",
    handlers=[RichHandler(markup=True)]
)

# Заголовки столбцов
CSV_HEADERS = [
    "user_id", 
    "question", 
    "context", 
    "answer", 
    "score_question", 
    "score_context", 
    "timestamp"
]

LOG_FILE = Path("qa_logs.csv")

# Создаем файл с заголовками, если его нет
if not LOG_FILE.exists():
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def full_log(logger: logging.Logger, where: str) -> None:
    tb = traceback.format_exc()
    logger.info(f"=== Exception in {where} ===")
    logger.info(tb)

def log_qa(user_id: int, question: str, context: str, answer: str, score_question: float, score_context: float) -> None:
    timestamp = datetime.now().isoformat()
    
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            user_id,
            question.replace("\n", " ").strip(),
            context.replace("\n", " ").strip(),
            answer.replace("\n", " ").strip(),
            score_question,
            score_context,
            timestamp
        ])
