from transformers import T5ForConditionalGeneration, GPT2Tokenizer
import logging

from qdrant import get_chunks, compare_embeddings
from summarize import summarize_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model")

model_name = "RussianNLP/FRED-T5-Summarizer"

model = T5ForConditionalGeneration.from_pretrained("RussianNLP/FRED-T5-Summarizer")
tokenizer = GPT2Tokenizer.from_pretrained("RussianNLP/FRED-T5-Summarizer", eos_token="</s>")

params = {
    "device": "cuda",
    "no_repeat_ngram_size": 2,
    "num_beams": 4,
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
