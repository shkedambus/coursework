import os

import fitz
import gdown
import requests
import telebot
from dotenv import load_dotenv
from telebot.types import KeyboardButton, Message, ReplyKeyboardMarkup

from chunking import (hybrid_chunking, merge_short_chunks,
                      split_text_into_chunks)
from generate_answer import answer_user_question
from logger import get_logger
from qdrant import delete_user_data, embedd_chunks, update_db

logger = get_logger("Telegram bot")

load_dotenv()
api_key = os.getenv("API_TOKEN")
bot = telebot.TeleBot(api_key)

ADMIN_USER_ID = os.getenv("ADMIN_USER_ID")
OUTPUT_FOLDER = "downloaded_pdfs"

def extract_pdf_structure(pdf_path, header_font_threshold=14):
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

def process_text(text):
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

def add_documents(message: Message):
    """
    Обрабатывает ссылку на папку в Google Drive, извлекает текст из PDF-документов, разбивает его на смысловые чанки, 
    создаёт эмбеддинги и загружает их в векторную базу данных для указанного пользователя.
    """
    user_id = message.from_user.id
    google_drive_url = message.text.strip()

    output_folder = "downloaded_pdfs"
    chunks = []

    status_message = bot.send_message(user_id, "⚙️ Обрабатываю документы...")

    try:
        # Скачиваем всю папку из Google Drive
        gdown.download_folder(url=google_drive_url, output=output_folder, quiet=False, use_cookies=False)

        for filename in os.listdir(output_folder):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(output_folder, filename)
                pdf_text = process_text(extract_pdf_structure(pdf_path))

            if pdf_text: # Если документы не оказались пустыми
                chunks.extend(split_text_into_chunks(text=pdf_text, chunk_size=128, overlap=12))
                # chunks.extend(merge_short_chunks(hybrid_chunking(text=pdf_text, chunk_size=512, overlap=50)))

        if chunks:
            logger.info(f"Создано {len(chunks)} новых чанков. Начинаю эмбеддинг...")
            embeddings = embedd_chunks(chunks)
            update_db(collection_name="rag", chunks=chunks, embeddings=embeddings, user_id=user_id)

            bot.edit_message_text(
                "✅ Документы успешно добавлены в базу знаний!",
                chat_id=user_id,
                message_id=status_message.message_id
            )
        else:
            bot.edit_message_text(
                "⚠️ Не удалось извлечь текст из документов или документы пустые.",
                chat_id=user_id,
                message_id=status_message.message_id
            )
        
    except requests.exceptions.MissingSchema as e:
        logger.error(f"Некорректная ссылка: {str(e)}")
        bot.edit_message_text(
            "❌ Некорректная ссылка. Убедитесь, что ссылка начинается с https://.",
            chat_id=user_id,
            message_id=status_message.message_id
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}", exc_info=True)

        error_message = "❌ Произошла непредвиденная ошибка. Попробуйте еще раз."
        if "Cannot retrieve the folder information from the link" in str(e):
            error_message = "❌ Не удалось получить доступ к Google Drive. Проверьте, что папка доступна по ссылке."

        bot.edit_message_text(
            error_message,
            chat_id=user_id,
            message_id=status_message.message_id
        )

    finally:
        for filename in os.listdir(output_folder):
            pdf_path = os.path.join(output_folder, filename)
            os.remove(pdf_path)

def generate_keyboard() -> ReplyKeyboardMarkup:
    """
    Создает постоянную клавиатуру с основными кнопками.
    """
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    keyboard.add(
        KeyboardButton("Добавить документы в базу"),
        KeyboardButton("Очистить базу документов"),
        KeyboardButton("Помощь")
    )
    return keyboard

# Приветственное сообщение
@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    username = message.from_user.first_name
    bot.reply_to(
        message, 
        f"Привет, {username}, я твой помощник! Пришли мне PDF документ - я добавлю его в базу. Или просто задай вопрос!", 
        reply_markup=generate_keyboard()
    )

# Обработка сообщений
@bot.message_handler(content_types=["text"])
def handle_text(message):
    user_id = message.from_user.id
    text = message.text.strip()
    
    if text == "Добавить документы в базу":
        msg = bot.send_message(user_id, "Введи ссылку на Google Drive с PDF документами.")
        bot.register_next_step_handler(msg, add_documents)
    elif text == "Очистить базу документов":
        delete_user_data(collection_name="rag", user_id=user_id)
        bot.send_message(user_id, "✅ База документов очищена!")
    elif text == "Помощь":
        send_welcome(message)
    else:
        # Обработка обычных вопросов
        status_message = bot.send_message(user_id, "⚙️ Думаю...")
        answer = answer_user_question(user_id, text)
        
        if answer == "Ошибка обработки запроса.":
            bot.edit_message_text(
                "❌ Что-то пошло не так. Попробуйте снова.",
                chat_id=user_id,
                message_id=status_message.message_id
            )
        else:
            bot.edit_message_text(
                answer,
                chat_id=user_id,
                message_id=status_message.message_id
            )
    
bot.infinity_polling() # Запуск бота
