from typing import List

import torch
from razdel import tokenize


def split_tokens_into_chunks(tokens: torch.Tensor, chunk_size: int, overlap: float = 0.0) -> List[torch.Tensor]:
    """
    Делит токены на чанки с возможным перекрытием.
    """
    stride = int(chunk_size * (1 - overlap))
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), stride) if len(tokens[i:i + chunk_size]) > 0]

def compute_prefix_ids(tokenizer, prefix: str) -> torch.Tensor:
    """
    Вычисляет токены префикса, если он есть.
    """
    if prefix.strip():
        return tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
    return torch.tensor([], dtype=torch.long)

def rouge_tokenizer(text: str) -> str:
    """
    Преобразует текст в токенизированную строку для оценки с помощью метрики ROUGE.

    Использует библиотеку razdel для токенизации текста на русском языке.
    Возвращает строку, где токены разделены пробелами - такой формат ожидается метрикой ROUGE.
    """
    tokens = [token.text for token in tokenize(text)]
    return " ".join(tokens)
