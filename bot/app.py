import os
import shutil
from pathlib import Path

import fitz
import gdown
import httpx
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters.command import Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext

from chunking import split_text_into_chunks

class AddDocs(StatesGroup):
    waiting_for_link = State()

from shared.logger import get_logger, log_qa
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

class Emoji:
    ERROR = "⚠️"
    SUCCESS = "✅"
    PROCESSING = "⚙️"
    THINKING = "🤔"
    DOCUMENT = "📄"
    DATABASE = "🗄️"
    WARNING = "❗"

class Messages:
    WELCOME = (
        f"Привет, я твой персональный ассистент!\n\n"
        "Я могу:\n"
        f"- {Emoji.DOCUMENT} Добавлять PDF-документы из Google Drive или файлов\n"
        f"- {Emoji.DATABASE} Искать информацию в добавленных документах\n"
        f"- {Emoji.WARNING} Очищать базу знаний при необходимости\n\n"
        "Просто отправьте мне документ или ссылку на папку Google Drive!"
    )
    
    ADD_DOCS = (
        f"{Emoji.DOCUMENT} Отправьте:\n"
        "- Ссылку на папку Google Drive с PDF\n"
        "- Или прикрепите PDF-файл напрямую\n\n"
    )

class Buttons:
    CANCEL = [InlineKeyboardButton(text="Отменить", callback_data="cancel")]
    ADD_DOCS = [InlineKeyboardButton(text="Добавить документы", callback_data="add_docs")]
    CLEAR_DB = [InlineKeyboardButton(text="Очистить базу", callback_data="clear_db")]
    
# Генерация inline кнопок
def get_main_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[Buttons.ADD_DOCS, Buttons.CLEAR_DB])

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

async def send_data(url: str, data: dict) -> dict:
    """
    Отправляет неблокирующие HTTP запросы.
    """
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            response = await client.post(url=url, json=data)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Ошибка соединения: {e}")
        return {"error": str(e)}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP ошибка: {e.response.status_code} - {e.response.text}")
        return {"error": str(e)}

# Приветственное сообщение
@dp.message(Command("start", "help"))
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer(   
        Messages.WELCOME,
        reply_markup=get_main_keyboard()
    )

# Кнопка отмены
@dp.callback_query(F.data == "cancel")
async def handle_cancel(callback: types.CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(
        f"{Emoji.WARNING} Действие отменено",
        reply_markup=get_main_keyboard()
    )

# Очистка базы
@dp.callback_query(F.data == "clear_db")
async def clear_db(callback: types.CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    url = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/clear"
    data = {"user_id": user_id}
    response = await send_data(url=url, data=data)
    error = response.get("error", "")
    if not error:
        await callback.message.answer(f"{Emoji.SUCCESS} База успешно очищена", reply_markup=InlineKeyboardMarkup(inline_keyboard=[Buttons.ADD_DOCS]))
    else:
        await callback.message.answer(f"{Emoji.ERROR} Ошибка при очистке базы", reply_markup=get_main_keyboard())

# Добавление документов в базу
@dp.callback_query(F.data == "add_docs")
async def handle_add_docs(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.answer(
        Messages.ADD_DOCS,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[Buttons.CANCEL])
    )
    await state.set_state(AddDocs.waiting_for_link)

# Обработчик для добавления документов в базу
@dp.message(AddDocs.waiting_for_link, F.text | F.document)
async def process_input(message: types.Message, state: FSMContext):
    if message.document:
        await process_document(message, state)
    else:
        await process_link(message, state)

# Добавление документов в базу через url на Google Drive
@dp.message(AddDocs.waiting_for_link, F.text)
async def process_link(message: types.Message, state: FSMContext):
    link = message.text.strip()
    user_id = message.from_user.id
    status_message = await message.answer(f"{Emoji.DOCUMENT} Обрабатываю документы...",
                                          reply_markup=InlineKeyboardMarkup(inline_keyboard=[Buttons.CANCEL])
    )
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
                f"{Emoji.ERROR} Не удалось извлечь текст из документов или документы пустые",
                reply_markup=get_main_keyboard()
            )
            return

        logger.info(f"Создано {len(chunks)} новых чанков. Начинаю эмбеддинг...")
        await status_message.edit_text(f"{Emoji.PROCESSING} Отправляю данные в базу, это может занять несколько минут...",
                                       reply_markup=InlineKeyboardMarkup(inline_keyboard=[Buttons.CANCEL])
        )

        url = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/upsert"
        data = {"user_id": user_id, "chunks": chunks}
        response = await send_data(url=url, data=data)

        await status_message.edit_text(
            f"{Emoji.SUCCESS} Документы добавлены в базу знаний!",
            reply_markup=get_main_keyboard()
        )
        await state.clear()

    except Exception as e:
        logger.error("Ошибка при добавлении документов", exc_info=True)
        await status_message.edit_text(
            f"{Emoji.ERROR} Не удалось добавить документы. Проверьте ссылку и попробуйте снова."
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# Добавление документов в базу напрямую из файла
@dp.message(AddDocs.waiting_for_link, F.document)
async def process_document(message: types.Message, state: FSMContext):
    if message.document.mime_type != "application/pdf":
        await message.answer(
            f"{Emoji.ERROR} Поддерживаются только PDF-файлы. Попробуйте снова"
        )
        return
    
    status_message = await message.answer(f"{Emoji.DOCUMENT} Обрабатываю документ...",
                                          reply_markup=InlineKeyboardMarkup(inline_keyboard=[Buttons.CANCEL]))
    user_id = message.from_user.id
    tmp_dir = Path(f"tmp_{user_id}")
    tmp_dir.mkdir(exist_ok=True)
    
    try:
        # Скачиваем файл
        file = await bot.get_file(message.document.file_id)
        file_path = tmp_dir / message.document.file_name
        await bot.download_file(file.file_path, destination=file_path)
        
        text = extract_pdf_structure(str(file_path))
        text = process_text(text)
        chunks = split_text_into_chunks(text=text, chunk_size=128, overlap=12)
        
        if not chunks:
            await status_message.edit_text(f"{Emoji.ERROR} Не удалось извлечь текст из документа",
                                           reply_markup=get_main_keyboard()
            )
            return
            
        await status_message.edit_text(f"{Emoji.PROCESSING} Отправляю данные в базу, это может занять несколько минут...",
                                       reply_markup=InlineKeyboardMarkup(inline_keyboard=[Buttons.CANCEL])
        )
        url = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/upsert"
        data = {"user_id": user_id, "chunks": chunks}
        response = await send_data(url=url, data=data)
        
        await status_message.edit_text(f"{Emoji.SUCCESS} Документ добавлен в базу знаний!",
                                       reply_markup=get_main_keyboard()
        )
        await state.clear()
        
    except Exception as e:
        logger.error("Ошибка обработки документа", exc_info=True)
        await status_message.edit_text(f"{Emoji.ERROR} Ошибка обработки документа")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# Обработка вопроса
@dp.message(F.text)
async def handle_question(message: types.Message):
    user_id = message.from_user.id
    query = message.text.strip()
    
    status_message = await message.answer(f"{Emoji.DATABASE} Получаю данные из базы...",
                                          reply_markup=InlineKeyboardMarkup(inline_keyboard=[Buttons.CANCEL]))

    try:
        url_1 = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/retrieve"
        data_1 = {"user_id": user_id, "query": query}
        response_1 = await send_data(url=url_1, data=data_1)
        chunks = response_1.get("chunks", [])

        if not chunks:
            await status_message.edit_text(
                "К сожалению, в предоставленном контексте нет информации по этому вопросу.",
                reply_markup=get_main_keyboard()
            )
            return
        
        context = " ".join(chunks)
        
        await status_message.edit_text(
            f"{Emoji.THINKING} Данные получены, думаю над ответом...",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[Buttons.CANCEL])
        )

        url_2 = f"http://{GENERATOR_HOST}:{GENERATOR_PORT}/summarize"
        data_2 = {"user_id": user_id, "question": query, "context": context}
        response_2 = await send_data(url=url_2, data=data_2)
        summary = response_2.get("summary", "")

        await status_message.edit_text(summary)

        url_3 = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/compare"
        data_3 = {"question": query, "context": context, "summary": summary}
        response_3 = await send_data(url=url_3, data=data_3)

        score_question = response_3.get("score_question", 0.0)
        score_context = response_3.get("score_context", 0.0)

        log_qa(
            user_id=user_id,
            question=query,
            context=context,
            answer=summary,
            score_question=score_question,
            score_context=score_context
        )

    except Exception as e:
        logger.error("Ошибка при обработке вопроса", exc_info=True)
        await status_message.edit_text(
            f"{Emoji.ERROR} Произошла ошибка при получении ответа. Попробуйте снова",
            reply_markup=get_main_keyboard()
        )

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
