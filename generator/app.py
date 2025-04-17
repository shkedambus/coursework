import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, GPT2Tokenizer
from summarize import summarize_text

from shared.logger import get_logger, full_log
logger = get_logger("generator/app.py")

class SummarizeRequest(BaseModel):
    user_id: int
    question: str
    context: str

class SummarizeResponse(BaseModel):
    summary: str

app = FastAPI(title="RAG Generator")

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
model = model.to(device)
model.eval()

def compute_prefix_ids(tokenizer, prefix: str) -> torch.Tensor:
    """
    Вычисляет токены префикса, если он есть.
    """
    if prefix.strip():
        return tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
    return torch.tensor([], dtype=torch.long)

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    # Подготовка префиксов для модели
    prefix = "<LM> Сократи текст.\n"
    prefix_ids = compute_prefix_ids(tokenizer=tokenizer, prefix=prefix)
    final_prefix = f"<LM> Ответь на вопрос:\n{req.question}.\nКонтекст:\n"
    final_prefix_ids = compute_prefix_ids(tokenizer=tokenizer, prefix=final_prefix)

    try:
        summary = summarize_text(
            article=req.context, 
            model_name=model_name, 
            tokenizer=tokenizer, 
            model=model, 
            prefix_ids=prefix_ids, 
            final_prefix_ids=final_prefix_ids, 
            params=model_params
        )
        return SummarizeResponse(summary=summary)

    except Exception as e:
        full_log(logger=logger, where="/summarize")
        # logger.error(f"Ошибка суммаризации: user_id={req.user_id}, question={req.question}, error={e}")
        return SummarizeResponse(summary="❌ Ошибка обработки запроса.")
