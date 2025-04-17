import os
import shutil
from pathlib import Path

import fitz
import gdown
import requests
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters.command import Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext

from chunking import split_text_into_chunks

class AddDocs(StatesGroup):
    waiting_for_link = State()

from shared.logger import get_logger
logger = get_logger("bot/app.py")

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
RETRIEVER_HOST = os.getenv("RETRIEVER_HOST")
RETRIEVER_PORT = os.getenv("RETRIEVER_PORT")
GENERATOR_HOST = os.getenv("GENERATOR_HOST")
GENERATOR_PORT = os.getenv("GENERATOR_PORT")

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

def extract_pdf_structure(pdf_path: str, header_font_threshold: int = 14) -> str:
    """
    Извлекает текст из PDF-файла, сохраняя структуру: заголовки, абзацы и списки.
    """
    doc = fitz.open(pdf_path)
    structured_text = []

    for page in doc:
        # Получаем словарное представление страницы
        page_dict = page.get_text("dict")
        
        # Проходим по каждому блоку (обычно это абзацы или отдельные текстовые блоки)
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # 0 означает текстовый блок
                block_lines = []
                for line in block["lines"]:
                    line_text = ""
                    max_font_size = 0
                    
                    # Объединяем все спаны в строку и определяем максимальный размер шрифта
                    for span in line["spans"]:
                        line_text += span["text"]
                        if span["size"] > max_font_size:
                            max_font_size = span["size"]
                    
                    # Если максимальный размер шрифта превышает порог, считаем строку заголовком
                    if max_font_size >= header_font_threshold:
                        # Добавляем пустые строки для визуального разделения
                        block_lines.append("\n" + line_text.strip() + "\n")
                    else:
                        block_lines.append(line_text.strip())
                
                # Объединяем строки блока в один абзац
                structured_text.append("\n".join(block_lines))
                # Добавляем дополнительный перевод строки между блоками
                structured_text.append("\n")
    
    return "\n".join(structured_text)

def process_text(text: str) -> str:
    """
    Собирает буллет-поинты в тексте в свои блоки.
    """
    lines = text.splitlines()
    structured_text = []
    i = 0
    n = len(lines)
    
    while i < n:
        line = lines[i].strip()
        if line == "•":
            # Собираем все последующие непустые строки в блок
            bullet_block = [line + " " + lines[i+1].strip() if i+1 < n else ""]
            i += 2
            while i < n and lines[i].strip():
                bullet_block.append(lines[i].strip())
                i += 1
            structured_text.append("\n".join(bullet_block))
        else:
            structured_text.append(line)
            i += 1
    
    return "\n".join(structured_text)

# Генерация кнопок
def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Добавить документы")],
            [KeyboardButton(text="Очистить базу")],
            [KeyboardButton(text="Помощь")]
        ],
        resize_keyboard=True,
        row_width=2
    )

# Приветственное сообщение
@dp.message(Command("start", "help"))
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "Привет! Я твой бот-помощник.\n\n"
        "- Пришли ссылку на папку Google Drive с PDF, чтобы добавить документы.\n"
        "- Нажми «Очистить базу», чтобы удалить всё.\n"
        "- Просто задай вопрос, и я отвечу на основе загруженных документов.",
        reply_markup=main_keyboard()
    )

# Очистка базы
@dp.message(F.text == "Очистить базу")
async def clear_db(message: types.Message):
    user_id = message.from_user.id
    url = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/clear"
    response = requests.post(url, json={"user_id": user_id})
    if response.ok:
        await message.answer("✅ База успешно очищена.")
    else:
        await message.answer("❌ Ошибка при очистке базы.")

# Добавление документов в базу
@dp.message(F.text == "Добавить документы")
async def cmd_add_docs(message: types.Message, state: FSMContext):
    await message.answer("Отправь ссылку на папку Google Drive с PDF‑файлами.")
    await state.set_state(AddDocs.waiting_for_link)

# Добавление документов в базу
@dp.message(AddDocs.waiting_for_link, F.text)
async def process_link(message: types.Message, state: FSMContext):
    link = message.text.strip()
    user_id = message.from_user.id
    status_message = await message.answer("⚙️ Обрабатываю документы...")
    tmp_dir = Path(f"tmp_{user_id}")
    tmp_dir.mkdir(exist_ok=True)

    try:
        # Скачиваем всю папку
        gdown.download_folder(link, output=str(tmp_dir), quiet=True, use_cookies=False)

        chunks = []
        for pdf_path in tmp_dir.glob("*.pdf"):
            text = extract_pdf_structure(str(pdf_path))
            text = process_text(text)
            chunks += split_text_into_chunks(text=text, chunk_size=128, overlap=12)

        if not chunks:
            await status_message.edit_text(
                "⚠️ Не удалось извлечь текст из документов или документы пустые."
            )
            return

        logger.info(f"Создано {len(chunks)} новых чанков. Начинаю эмбеддинг...")
        url = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/upsert"
        response = requests.post(url, json={"user_id": user_id, "chunks": chunks})
        response.raise_for_status()

        await status_message.edit_text(
            "✅ Документы добавлены в базу знаний!"
        )

    except Exception as e:
        logger.error("Ошибка при добавлении документов", exc_info=True)
        await status_message.edit_text(
            "❌ Не удалось добавить документы. Проверь ссылку и попробуй снова."
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        await state.clear()

# Обработка вопроса
@dp.message(F.text)
async def handle_question(message: types.Message):
    user_id = message.from_user.id
    query = message.text.strip()
    
    if query in ["Добавить документы", "Очистить базу", "Помощь"]:
        return
    
    status_message = await message.answer("⚙️ Думаю...")

    try:
        url_1 = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/retrieve"
        response_1 = requests.post(url_1, json={"user_id": user_id, "query": query})
        response_1.raise_for_status()
        chunks = response_1.json().get("chunks", [])

        if not chunks:
            await status_message.edit_text(
                "К сожалению, в предоставленном контексте нет информации по этому вопросу."
            )
            return

        url_2 = f"http://{GENERATOR_HOST}:{GENERATOR_PORT}/summarize"
        response_2 = requests.post(url_2, json={"user_id": user_id, "question": query, "context": " ".join(chunks)})
        response_2.raise_for_status()
        answer = response_2.json().get("summary", "")

        await status_message.edit_text(answer or "❌ Пустой ответ.")

    except Exception as e:
        logger.error("Ошибка при обработке вопроса", exc_info=True)
        await status_message.edit_text(
            "❌ Произошла ошибка при получении ответа. Попробуй снова."
        )

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
