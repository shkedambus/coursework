from datetime import datetime

import torch
from transformers import GPT2Tokenizer, T5ForConditionalGeneration

from qdrant import compare_embeddings, get_relevant_chunks
from summarize import summarize_text
from utils import compute_prefix_ids

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

from logger import get_logger

logger = get_logger("RAG model")

model_name = "RussianNLP/FRED-T5-Summarizer"
model = T5ForConditionalGeneration.from_pretrained("RussianNLP/FRED-T5-Summarizer")
tokenizer = GPT2Tokenizer.from_pretrained("RussianNLP/FRED-T5-Summarizer", eos_token="</s>")

model_params = {
    "device": "cuda",
    "no_repeat_ngram_size": 4,
    "num_beams": 5,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.7,
    "overlap": 0.2,
}

device = model_params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
model = model.to("cpu")
model.eval()

def answer_user_question(user_id: int, question: str) -> str:
    """
    Генерирует ответ на вопрос пользователя, опираясь на контекст из базы документов
    """
    try:
        retrieved_chunks = get_relevant_chunks(collection_name="rag", query=question, user_id=user_id)
    except Exception as e:
        logger.error(f"Ошибка при получении чанков: user_id={user_id}, question={question}, error={e}")
        return "В данный момент система не может обработать ваш запрос."

    context = " ".join(retrieved_chunks)

    # Подготовка префиксов для модели
    prefix = "<LM> Сократи текст.\n"
    prefix_ids = compute_prefix_ids(tokenizer=tokenizer, prefix=prefix)
    final_prefix = f"<LM> Ответь на вопрос:\n{question}.\nКонтекст:\n"
    final_prefix_ids = compute_prefix_ids(tokenizer=tokenizer, prefix=final_prefix)

    response = "К сожалению, в предоставленном контексте нет информации по этому вопросу."
    quality_score_question = None
    quality_score_context = None

    if context:
        try:
            model.to(device)
            response = summarize_text(
                article=context, 
                model_name=model_name, 
                tokenizer=tokenizer, 
                model=model, 
                prefix_ids=prefix_ids, 
                final_prefix_ids=final_prefix_ids, 
                params=model_params
            )
            model.to("cpu")
        except Exception as e:
            logger.error(f"Ошибка суммаризации: user_id={user_id}, question={question}, error={e}")
            return "Ошибка обработки запроса."

        # Проверка качества суммаризации через вычисление cosine similarity
        try:
            quality_score_question = round(float(compare_embeddings(predictions=[response], references=[question])[0][0]), 4)
            quality_score_context = round(float(compare_embeddings(predictions=[response], references=[context])[0][0]), 4)
        except Exception as e:
            logger.error(f"Ошибка вычисления quality score: user_id={user_id}, question={question}, error={e}")
            quality_score_context = None
            quality_score_question = None

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "question": question,
        "context": context if context else "Контекст отсутствует",
        "response": response,
        "quality_score_question": quality_score_question,
        "quality_score_context": quality_score_context
    }
    logger.info(f"Query report: {log_data}")

    return response
