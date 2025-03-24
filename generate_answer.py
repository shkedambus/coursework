from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

from qdrant import get_chunks, compare_embeddings
from summarize import summarize_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model")

model_name = "csebuetnlp/mT5_m2o_russian_crossSum"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, legacy=False, use_fast=False)

params = {
    "device": "cuda",
    "max_length": 512,
    "min_length": 30,
    "no_repeat_ngram_size": 4,
    "add_special_tokens": False,
    "num_beams": 10,
    "max_new_tokens": 200,
    "top_p": 0.7
}

def answer_user_question(user_id, question):
    retrieved_chunks = get_chunks("main", question, user_id)
    context = "\n".join(retrieved_chunks)

    response = "К сожалению, в предоставленном контексте нет информации по этому вопросу."
    if context:
        response = summarize_text(article=context, model_name=model_name, tokenizer=tokenizer, model=model, params=params)

    score = round(float(compare_embeddings(predictions=[response], references=[context])[0][0]), 4)
    
    with open(file="report.txt", mode="a", encoding="utf-8") as f:
        f.write(f"Вопрос:\n{question}\n")
        if not context:
            context = "Контекст отсутствует."
        f.write(f"Контекст:\n{context}\n")
        f.write(f"Ответ:\n{response}\n")
        f.write(f"Скор: {score}\n\n")
        f.close()

    return response
