import os
import uuid
from dotenv import load_dotenv

import telebot
import requests
import fitz

from chunking import hybrid_chunking, merge_short_chunks
from qdrant import embedd_chunks, update_db, inspect_collection, clear_collection
from generate_answer import answer_user_question

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Bot")

load_dotenv()
api_key = os.getenv("API_TOKEN")

bot = telebot.TeleBot(api_key)

my_user_id = 362592209

# Функция для скачивания PDF документа по ссылке и сохранения его в файл
def download_pdf(url, save_path):
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers, stream=True)
    
    # Сохраняем файл
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return save_path

# Функция для извлечения текста из PDF документа
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

# Функция для обработки PDF документов
@bot.message_handler(content_types=["document"])
def handle_documents(message):
    user_id = message.from_user.id
    documents = message.document if isinstance(message.document, list) else [message.document]

    # bot.send_message(user_id, f"Получено {len(documents)} документа(ов), загружаю...")

    for doc in documents:
        file_id = doc.file_id
        file_info = bot.get_file(file_id)
        file_url = f"https://api.telegram.org/file/bot{api_key}/{file_info.file_path}"

        unique_name = f"pdf_{uuid.uuid4().hex}.pdf"
        pdf_path = os.path.join("downloads", unique_name)

        # Скачиваем PDF
        pdf_path = download_pdf(file_url, pdf_path)
        if not pdf_path:
            bot.send_message(user_id, f"Не удалось скачать {doc.file_name}.")
            continue

        # Извлекаем текст
        pdf_text = extract_text_from_pdf(pdf_path)
        
        if not pdf_text.strip():
            bot.send_message(user_id, f"Файл {doc.file_name} пуст.")
            continue

        # Разбиваем текст на чанки, векторизуем и загружаем в Qdrant
        chunks = hybrid_chunking(pdf_text)
        chunks = merge_short_chunks(chunks)
        embeddings = embedd_chunks(chunks)
        update_db("main", chunks, embeddings, user_id)

        bot.send_message(user_id, f"{doc.file_name} добавлен в базу знаний!")

# Приветственное сообщение
@bot.message_handler(commands=["start", "help", "cat", "reset"])
def send_welcome(message):
    user_id = message.from_user.id
    command = message.text
    if command == "/cat":
        if user_id == my_user_id:
            inspect_collection("main", limit=100)
        else:
            bot.send_message(user_id, "Извините, вы не можете использовать эту комманду.")
    elif command == "/reset":
        if user_id == my_user_id:
            clear_collection("main")
        else:
            bot.send_message(user_id, "Извините, вы не можете использовать эту комманду.")
    else:
        bot.send_message(user_id, "Привет! Пришли мне PDF-документ - я добавлю его в базу. Или просто задай вопрос!")

# Функция для обработки сообщений (вопросов)
@bot.message_handler(content_types=["text"])
def handle_text(message):
    user_id = message.from_user.id
    question = message.text.strip()

    answer = answer_user_question(user_id, question)
    bot.send_message(user_id, answer)
    
bot.infinity_polling()  # запуск бота