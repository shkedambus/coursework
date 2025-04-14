import os
import time

import fitz
import gdown

from chunking import hybrid_chunking, merge_short_chunks, split_text_into_chunks
from generate_answer import answer_user_question
from logger import get_logger
from qdrant import (clear_collection, delete_user_data, embedd_chunks,
                    inspect_collection, update_db)

logger = get_logger("Console app")

import nltk

from qdrant import embedding_model

# Нужно скачать 1 раз
# nltk.download('punkt')

ADMIN_USER_ID = 362592209

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
    Объединяет буллет-поинты в один блок.
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

def main_menu() -> None:
    print("\n(1) Загрузить документы (через ссылку на Google Drive)")
    print("(2) Задать вопрос")
    print("(3) Очистить общую базу документов")
    print("(4) Очистить свою базу документов")
    print("(5) Осмотреть общую базу документов")
    print("(6) Осмотреть свою базу документов\n")

    while True:
        option = input("Введите нужную цифру: ")
        time.sleep(1)
        if option in ["1", "2", "3", "4", "5", "6"]:
            break
        else:
            print("Попробуйте еще раз.\n")
            time.sleep(1)

    if option == "1":
        google_drive_url = input("Введите ссылку на Google Drive: ")
        output_folder = "downloaded_pdfs"
        chunks = []

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
                update_db(collection_name="rag", chunks=chunks, embeddings=embeddings, user_id=ADMIN_USER_ID)
            
        except Exception as e:
            print("Произошла ошибка:", e, end="\n")
        finally:
            for filename in os.listdir(output_folder):
                pdf_path = os.path.join(output_folder, filename)
                os.remove(pdf_path)

    elif option == "2":
        question = input("Что вас интересует?\n")
        
        answer = answer_user_question(user_id=ADMIN_USER_ID, question=question)
        print(answer)
    elif option == "3":
        clear_collection(collection_name="rag")
    elif option == "4":
        delete_user_data(collection_name="rag", user_id=ADMIN_USER_ID)
    elif option == "5":
        inspect_collection(collection_name="rag")
    elif option == "6":
        inspect_collection(collection_name="rag", user_id=ADMIN_USER_ID)

    time.sleep(1)
    main_menu()

print("\nПривет! Пришли мне ссылку на PDF-документ(ы) - я добавлю его(их) в базу. Или просто задай вопрос!")
main_menu()
