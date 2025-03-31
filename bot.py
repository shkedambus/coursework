import os
import uuid

import fitz
import gdown
import requests
import telebot
from dotenv import load_dotenv
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from chunking import hybrid_chunking, merge_short_chunks
from generate_answer import answer_user_question
from logger import get_logger
from qdrant import delete_user_data, embedd_chunks, update_db

logger = get_logger("Telegram bot")

load_dotenv()
api_key = os.getenv("API_TOKEN")
bot = telebot.TeleBot(api_key)

ADMIN_USER_ID = os.getenv("ADMIN_USER_ID")
OUTPUT_FOLDER = "downloaded_pdfs"

def download_pdf(url: str, save_path: str) -> str:
    """
    Скачивает PDF документ по ссылке и сохраняет его в файл.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True)
    
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return save_path

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает текст из PDF документа.
    """
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def generate_markup() -> InlineKeyboardMarkup:
    """
    Формирует меню из кнопок.
    """
    markup = InlineKeyboardMarkup()
    markup.row_width = 3
    markup.add(
            InlineKeyboardButton("Добавить документы", callback_data="cb_add_documents"),
            InlineKeyboardButton("Очистить документы", callback_data="cb_delete_documents"),
        )
    return markup

def add_documents(message: Message):
    """
    Обрабатывает ссылку на папку в Google Drive, извлекает текст из PDF-документов, разбивает его на смысловые чанки, 
    создаёт эмбеддинги и загружает их в векторную базу данных для указанного пользователя.
    """
    user_id = message.from_user.id
    google_drive_url = message.text.strip()

    pdf_text = ""

    try:
        # Скачиваем всю папку из Google Drive
        gdown.download_folder(url=google_drive_url, output=OUTPUT_FOLDER, quiet=False, use_cookies=False)

        for filename in os.listdir(OUTPUT_FOLDER):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(OUTPUT_FOLDER, filename)
                pdf_text += extract_text_from_pdf(pdf_path)

        if pdf_text: # Если документы не оказались пустыми
            chunks = hybrid_chunking(pdf_text)
            chunks = merge_short_chunks(chunks)
            embeddings = embedd_chunks(chunks)
            update_db(collection_name="rag", chunks=chunks, embeddings=embeddings, user_id=user_id)
        
    except Exception as e:
        logger.info("Произошла ошибка:", e, end="\n")
    finally:
        for filename in os.listdir(OUTPUT_FOLDER):
            pdf_path = os.path.join(OUTPUT_FOLDER, filename)
            os.remove(pdf_path)

# Обработка прикрепления PDF документов
@bot.message_handler(content_types=["document"])
def handle_documents(message):
    user_id = message.from_user.id
    documents = message.document if isinstance(message.document, list) else [message.document]

    logger.info(f"Получено {len(documents)} документа(ов), загружаю...")

    for doc in documents:
        try:
            file_id = doc.file_id
            file_info = bot.get_file(file_id)
            file_url = f"https://api.telegram.org/file/bot{api_key}/{file_info.file_path}"

            unique_name = f"pdf_{uuid.uuid4().hex}.pdf"
            pdf_path = os.path.join(OUTPUT_FOLDER, unique_name)

            # Скачиваем PDF
            pdf_path = download_pdf(file_url, pdf_path)
            if not pdf_path:
                logger.info(f"Не удалось скачать {doc.file_name}.")
                continue

            # Извлекаем текст
            pdf_text = extract_text_from_pdf(pdf_path)
            
            if not pdf_text.strip():
                logger.info(f"Файл {doc.file_name} пуст.")
                continue

            # Разбиваем текст на чанки, векторизуем и загружаем в Qdrant
            chunks = hybrid_chunking(pdf_text)
            chunks = merge_short_chunks(chunks)
            embeddings = embedd_chunks(chunks)
            update_db(collection_name="rag", chunks=chunks, embeddings=embeddings, user_id=user_id)

        except Exception as e:
            logger.info("Произошла ошибка при загрузке {doc.file_name}:", e, end="\n")
            continue

    for filename in os.listdir(OUTPUT_FOLDER):
        pdf_path = os.path.join(OUTPUT_FOLDER, filename)
        os.remove(pdf_path)
    
    bot.send_message(user_id, "Документы были успешно добавлены в базу знаний!")

# Приветственное сообщение
@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    username = message.from_user.first_name
    bot.reply_to(message, f"Привет, {username}, я твой помощник! Пришли мне PDF документ - я добавлю его в базу. Или просто задай вопрос!", reply_markup=generate_markup())

# Обработка сообщений (вопросов)
@bot.message_handler(content_types=["text"])
def handle_text(message):
    user_id = message.from_user.id
    question = message.text.strip()

    answer = answer_user_question(user_id, question)
    bot.send_message(user_id, answer)

# Обработка кнопок меню
@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    user_id = call.from_user.id

    if call.data == "cb_add_documents":
        msg = bot.send_message(user_id, "Введи ссылку на Google Drive с PDF документами.")
        bot.register_next_step_handler(msg, add_documents)
    elif call.data == "cb_clear_documents":
        delete_user_data(collection_name="rag", user_id=user_id)
    
bot.infinity_polling() # Запуск бота
