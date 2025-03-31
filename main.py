import os
import time

import fitz
import gdown

from chunking import hybrid_chunking, merge_short_chunks
from generate_answer import answer_user_question
from qdrant import (clear_collection, embedd_chunks, inspect_collection,
                    update_db)

from logger import get_logger
logger = get_logger("Console app")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def main_menu():
    print("\n(1) Загрузить документы (через ссылку на Google Drive).")
    print("(2) Задать вопрос.")
    print("(3) Очистить базу документов.")
    print("(4) Осмотреть базу документов.\n")

    while True:
        option = int(input("Введите нужную цифру: "))
        time.sleep(1)
        if 1 <= option <= 4:
            break
        else:
            print("Попробуйте еще раз.\n")
            time.sleep(1)

    if option == 1:
        google_drive_url = input("Введите ссылку на Google Drive: ")
        output_folder = "downloaded_pdfs"
        pdf_text = ""

        try:
            # Скачиваем всю папку из Google Drive
            gdown.download_folder(url=google_drive_url, output=output_folder, quiet=False, use_cookies=False)

            for filename in os.listdir(output_folder):
                if filename.lower().endswith(".pdf"):
                    pdf_path = os.path.join(output_folder, filename)
                    pdf_text += extract_text_from_pdf(pdf_path)

            if pdf_text: # Если документы не оказались пустыми
                chunks = hybrid_chunking(pdf_text)
                chunks = merge_short_chunks(chunks)
                embeddings = embedd_chunks(chunks)
                update_db(collection_name="rag", chunks=chunks, embeddings=embeddings, user_id=0)
            
        except Exception as e:
            print("Произошла ошибка:", e, end="\n")
        finally:
            for filename in os.listdir(output_folder):
                pdf_path = os.path.join(output_folder, filename)
                os.remove(pdf_path)

    elif option == 2:
        question = input("Что вас интересует?\n")
        answer = answer_user_question(user_id=0, question=question)
        print(answer)
    elif option == 3:
        clear_collection(collection_name="rag")
    elif option == 4:
        inspect_collection(collection_name="rag")

    time.sleep(1)
    main_menu()

print("\nПривет! Пришли мне ссылку на PDF-документ(ы) - я добавлю его(их) в базу. Или просто задай вопрос!")
main_menu()
