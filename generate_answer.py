import torch
from transformers import GPT2Tokenizer, T5ForConditionalGeneration

from qdrant import compare_embeddings, get_chunks
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
    "no_repeat_ngram_size": 2,
    "num_beams": 4,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.7,
    "overlap": 0.2,
    "prefix": "",
}

device = model_params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

prefix_ids = compute_prefix_ids(tokenizer=tokenizer, prefix=model_params.get("prefix", ""))

def answer_user_question(user_id: int, question: str) -> str:
    """
    Генерирует ответ на вопрос пользователя, опираясь на контекст из базы документов
    """
    retrieved_chunks = get_chunks(collection_name="rag", query=question, user_id=user_id)
    context = " ".join(retrieved_chunks)

    response = "К сожалению, в предоставленном контексте нет информации по этому вопросу."
    if context:
        response = summarize_text(article=context, model_name=model_name, tokenizer=tokenizer, model=model, prefix_ids=prefix_ids, params=model_params)

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
