import os
import shutil
from pathlib import Path
from typing import List

import fitz
import gdown
import httpx
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters.command import Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext

from chunking import split_text_into_chunks, extract_pdf_structure, process_bullet_points

from shared.logger import get_logger, log_qa
logger = get_logger("bot/app.py")

class AddDocs(StatesGroup):
    waiting_for_url = State()

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
RETRIEVER_HOST = os.getenv("RETRIEVER_HOST")
RETRIEVER_PORT = os.getenv("RETRIEVER_PORT")
GENERATOR_HOST = os.getenv("GENERATOR_HOST")
GENERATOR_PORT = os.getenv("GENERATOR_PORT")
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID"))

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
    FOLDER = "📂"
    ADD = "📤"
    CLEAR = "🗑️"
    SMILE = "🙃"

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
    CANCEL = InlineKeyboardButton(text="Отменить", callback_data="cancel")
    ADD_DOCS = InlineKeyboardButton(text=f"{Emoji.ADD} Добавить", callback_data="add_docs")
    CLEAR_DB = InlineKeyboardButton(text=f"{Emoji.CLEAR} Очистить", callback_data="clear_db")
    LIST_FILES = InlineKeyboardButton(text=f"{Emoji.FOLDER} Мои файлы", callback_data="list_files")
    
# Генерация inline кнопок
def main_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [Buttons.ADD_DOCS, Buttons.CLEAR_DB, Buttons.LIST_FILES]
    ])

async def send_data(url: str, data: dict) -> dict:
    """
    Отправляет неблокирующие HTTP запросы.
    """
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            response = await client.post(url=url, json=data)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Ошибка соединения: {e}")
        return {"error": str(e)}

# Приветственное сообщение
@dp.message(Command("start", "help"))
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer(   
        Messages.WELCOME,
        reply_markup=main_menu()
    )

# Кнопка отмены
@dp.callback_query(F.data == "cancel")
async def handle_cancel(callback: types.CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(f"{Emoji.WARNING} Действие отменено")
    await asyncio.sleep(3)
    try:
        await callback.message.delete()
    except:
        pass

# Полная очистка базы данных для всех пользователей (admin only)
@dp.message(Command("drop"))
async def cmd_drop(message: types.Message, state: FSMContext):
    await state.clear()
    user_id = message.from_user.id
    if (user_id == ADMIN_USER_ID):
        url = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/drop"
        data = {"user_id": user_id}
        response = await send_data(url=url, data=data)
        success = response.get("success", False)
        if success:
            await message.answer(f"{Emoji.SUCCESS} База данных полностью очищена")
        else:
            await message.answer(f"{Emoji.ERROR} Ошибка при очистке базы")
    else:
        await message.answer(f"{Emoji.WARNING} Эта комманда доступна только для администратора")

# Просмотр загруженных файлов
@dp.callback_query(F.data == "list_files")
async def list_files(callback: types.CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id

    status_message = await callback.message.answer(f"{Emoji.DATABASE} Получаю данные из базы")

    url = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/list_files"
    data = {"user_id": user_id}
    response = await send_data(url=url, data=data)

    file_names = response.get("file_names")

    if not file_names:
        await status_message.edit_text(f"{Emoji.DATABASE} В вашей базе нет загруженных файлов")
        await callback.answer()
        return

    builder = InlineKeyboardBuilder()
    for i, file_name in enumerate(file_names):
        safe_name = file_name
        if len(file_name) > 30:
            safe_name = file_name[:30] + "..."

        builder.row(InlineKeyboardButton(
            text=safe_name,
            callback_data=f"delete_file_{i}"
        ))
    builder.row(InlineKeyboardButton(text="Отменить", callback_data="cancel"))

    await state.update_data(files_mapping={f"delete_file_{i}": file_name for i, file_name in enumerate(file_names)})

    # Один столбец
    builder.adjust(1)
    markup = builder.as_markup()

    await status_message.edit_text(
        f"{Emoji.FOLDER} Ваши файлы. Нажмите на тот, который хотите удалить:",
        reply_markup=markup
    )
    await callback.answer()

# Удаление выбранного файла
@dp.callback_query(F.data.startswith("delete_file_"))
async def cmd_delete_file(callback: types.CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    data = await state.get_data()
    file_name = data["files_mapping"].get(callback.data)

    await callback.message.edit_text(f"{Emoji.PROCESSING} Удаляю {file_name} из базы")

    url = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/delete_file"
    data = {"user_id": user_id, "file_name": file_name}
    response = await send_data(url=url, data=data)

    success = response.get("success")

    if success:
        message_text = f"{Emoji.SUCCESS} Файл {file_name} удалён"
    else:
        message_text = f"{Emoji.ERROR} Ошибка при удалении файла"

    await callback.message.edit_text(message_text)
    await callback.answer()

# Очистка базы
@dp.callback_query(F.data == "clear_db")
async def clear_db(callback: types.CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    url = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/clear"
    data = {"user_id": user_id}
    response = await send_data(url=url, data=data)

    success = response.get("success")
    empty = response.get("empty")

    if success:
        message_text = f"{Emoji.CLEAR} База данных пуста"
        if not empty:
            message_text = f"{Emoji.SUCCESS} База успешно очищена"
        await callback.message.answer(message_text)
    else:
        await callback.message.answer(f"{Emoji.ERROR} Ошибка при очистке базы")

    await callback.answer()

# Добавление документов в базу
@dp.callback_query(F.data == "add_docs")
async def handle_add_docs(callback: types.CallbackQuery, state: FSMContext):
    add_docs_msg = await callback.message.answer(
        Messages.ADD_DOCS,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[Buttons.CANCEL]])
    )
    await state.set_state(AddDocs.waiting_for_url)
    await state.update_data(add_docs_msg_id=add_docs_msg.message_id)
    await callback.answer()

# Обработчик для добавления документов в базу
@dp.message(AddDocs.waiting_for_url, F.text | F.document)
async def process_input(message: types.Message, state: FSMContext):
    data = await state.get_data()
    add_docs_msg_id = data.get("add_docs_msg_id")
    
    # Удаляем старое сообщение с инструкцией
    if add_docs_msg_id:
        try:
            await bot.delete_message(chat_id=message.chat.id, message_id=add_docs_msg_id)
        except:
            pass

    if message.document:
        await handle_file(message, state)
    else:
        await handle_url(message, state)

# Добавление документов в базу через url на Google Drive
@dp.message(AddDocs.waiting_for_url, F.text)
async def handle_url(message: types.Message, state: FSMContext):
    link = message.text.strip()
    user_id = message.from_user.id

    status_message = await message.answer(f"{Emoji.DOCUMENT} Обрабатываю документы",
                                          reply_markup=InlineKeyboardMarkup(inline_keyboard=[[Buttons.CANCEL]])
    )
    await message.delete()

    tmp_dir = Path(f"tmp_{user_id}")
    tmp_dir.mkdir(exist_ok=True)

    try:
        # Скачиваем всю папку
        gdown.download_folder(link, output=str(tmp_dir), quiet=True, use_cookies=False)

        chunks = []
        file_names = []
        for pdf_path in tmp_dir.glob("*.pdf"):
            text = extract_pdf_structure(str(pdf_path))
            text = process_bullet_points(text)

            file_name = str(pdf_path)
            file_name = file_name.lstrip(f"tmp_{user_id}/")

            for chunk in split_text_into_chunks(text=text, chunk_size=128, overlap=12):
                chunks.append(chunk)
                file_names.append(file_name)

            pdf_path.unlink(missing_ok=True)

        if not chunks:
            await status_message.edit_text(
                f"{Emoji.ERROR} Не удалось извлечь текст из документов или документы пустые")
            return

        logger.info(f"Создано {len(chunks)} новых чанков. Начинаю эмбеддинг")
        await status_message.edit_text(f"{Emoji.PROCESSING} Отправляю данные в базу, это может занять несколько минут",
                                       reply_markup=InlineKeyboardMarkup(inline_keyboard=[[Buttons.CANCEL]])
        )

        url = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/upsert"
        data = {"user_id": user_id, "chunks": chunks, "file_names": file_names}
        response = await send_data(url=url, data=data)

        await status_message.edit_text(
            f"{Emoji.SUCCESS} Документы добавлены в базу знаний!")
        await state.clear()

    except Exception as e:
        logger.error("Ошибка при добавлении документов", exc_info=True)
        await status_message.edit_text(f"{Emoji.ERROR} Не удалось добавить документы. Проверьте ссылку и попробуйте снова.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# Добавление документов в базу напрямую из файла
@dp.message(AddDocs.waiting_for_url, F.document)
async def handle_file(message: types.Message, state: FSMContext):
    if message.document.mime_type != "application/pdf":
        await message.answer(f"{Emoji.ERROR} Поддерживаются только PDF-файлы. Попробуйте снова")
        return
    
    user_id = message.from_user.id
    tmp_dir = Path(f"tmp_{user_id}")
    tmp_dir.mkdir(exist_ok=True)
    
    status_message = await message.answer(f"{Emoji.DOCUMENT} Обрабатываю документ",
                                          reply_markup=InlineKeyboardMarkup(inline_keyboard=[[Buttons.CANCEL]]))
    await message.delete()
    
    try:
        file = await bot.get_file(message.document.file_id)
        file_path = tmp_dir / message.document.file_name
        await bot.download_file(file.file_path, destination=file_path)
        
        text = extract_pdf_structure(str(file_path))
        text = process_bullet_points(text)

        if not text:
            await status_message.edit_text(
                f"{Emoji.ERROR} Не удалось извлечь текст из документов или документы пустые")
            return
    
        chunks = split_text_into_chunks(text=text, chunk_size=128, overlap=12)
        file_names = [message.document.file_name] * len(chunks)
            
        await status_message.edit_text(f"{Emoji.PROCESSING} Отправляю данные в базу, это может занять несколько минут",
                                       reply_markup=InlineKeyboardMarkup(inline_keyboard=[[Buttons.CANCEL]])
        )
        url = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/upsert"
        data = {"user_id": user_id, "chunks": chunks, "file_names": file_names}
        response = await send_data(url=url, data=data)
        
        await status_message.edit_text(f"{Emoji.SUCCESS} Документ добавлен в базу знаний!")
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
    
    status_message = await message.answer(f"{Emoji.DATABASE} Получаю данные из базы",
                                          reply_markup=InlineKeyboardMarkup(inline_keyboard=[[Buttons.CANCEL]]))

    try:
        url_1 = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/retrieve"
        data_1 = {"user_id": user_id, "query": query}
        response_1 = await send_data(url=url_1, data=data_1)
        chunks = response_1.get("chunks")

        if not chunks:
            await status_message.edit_text("К сожалению, в предоставленном контексте нет информации по этому вопросу")
            return
        
        context = " ".join(chunks)
        
        await status_message.edit_text(
            f"{Emoji.THINKING} Данные получены, думаю над ответом",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[[Buttons.CANCEL]])
        )

        url_2 = f"http://{GENERATOR_HOST}:{GENERATOR_PORT}/summarize"
        data_2 = {"user_id": user_id, "question": query, "context": context}
        response_2 = await send_data(url=url_2, data=data_2)
        summary = response_2.get("summary")

        await status_message.edit_text(summary)
        await status_message.answer(f"{Emoji.SMILE} Если у вас еще остались вопросы, буду рад помочь!", reply_markup=main_menu())

        url_3 = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}/compare"
        data_3 = {"question": query, "context": context, "summary": summary}
        response_3 = await send_data(url=url_3, data=data_3)

        score_question = response_3.get("score_question")
        score_context = response_3.get("score_context")

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
        await status_message.edit_text(f"{Emoji.ERROR} Произошла ошибка при получении ответа. Попробуйте снова")

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
