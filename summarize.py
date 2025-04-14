import torch
from tqdm import tqdm

from utils import split_tokens_into_chunks


def summarize_with_rugpt3(article: str, model_name: str, tokenizer, model, prefix_ids: torch.Tensor, params: dict) -> str:
    """
    Выполняет итеративную суммаризацию длинного текста (для модели rugpt3).

    Текст разбивается на части с учётом ограничения модели на максимальное число токенов.
    К каждому чанку при необходимости добавляется префикс.
    Для каждого чанка генерируется краткое содержание, которое затем объединяется и проходит повторную суммаризацию.
    """
    input_ids = tokenizer(
        article,
        return_tensors="pt",
        padding=False,
        truncation=False,   
        add_special_tokens=params.get("add_special_tokens", True)
    )["input_ids"]

    # Токены текста
    tokens = input_ids.squeeze(0)

    # Лимит модели по количеству токенов
    max_token_count = getattr(model.config, "max_position_embeddings", 512) // 2

    # Разбиваем текст на чанки
    prefix_length = len(prefix_ids)
    chunk_size = max_token_count - prefix_length
    chunks = split_tokens_into_chunks(tokens=tokens, chunk_size=chunk_size, overlap=params.get("overlap", 0))

    # Суммаризуем каждый чанк
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        all_output_ids = []
        for chunk_ids in tqdm(chunks, desc="Summarizing chunks"):
            max_new_tokens = max(int(len(chunk_ids) * 0.35), 200)
            min_length = max(int(len(chunk_ids) * 0.15), 50)

            eos_id = torch.tensor([tokenizer.sep_token_id], dtype=chunk_ids.dtype, device=chunk_ids.device)

            if prefix_length > 0:
                chunk_ids = torch.cat((prefix_ids, chunk_ids), dim=0)
            chunk_ids = torch.cat((chunk_ids, eos_id), dim=0)
            chunk_ids = chunk_ids.unsqueeze(0).to(device)

            output_ids = model.generate(
                input_ids=chunk_ids,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                no_repeat_ngram_size=params.get("no_repeat_ngram_size", 2),
                num_beams=params.get("num_beams", 4),
                do_sample=params.get("do_sample", False),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.9),
            )[0]

            all_output_ids.append(output_ids)

    # Декодируем все чанки за раз
    decoded_chunks = tokenizer.batch_decode(all_output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    for i in range(len(decoded_chunks)):
        decoded_chunks[i] = decoded_chunks[i].split(tokenizer.sep_token)[1]
        decoded_chunks[i] = decoded_chunks[i].split(tokenizer.eos_token)[0]

    if len(decoded_chunks) > 1:
        new_article = " ".join(decoded_chunks)
        summary = summarize_with_rugpt3(article=new_article, model_name=model_name, tokenizer=tokenizer, model=model, prefix_ids=prefix_ids, params=params)
    else:
        summary = decoded_chunks[0]

    return summary

def summarize_text(article: str, model_name: str, tokenizer, model, prefix_ids: torch.Tensor, final_prefix_ids: torch.Tensor, params: dict) -> str:
    """
    Выполняет итеративную суммаризацию длинного текста.

    Текст разбивается на части с учётом ограничения модели на максимальное число токенов.
    К каждому чанку при необходимости добавляется префикс.
    Для каждого чанка генерируется краткое содержание, которое затем объединяется и проходит повторную суммаризацию.
    """
    input_ids = tokenizer(
        article,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=params.get("add_special_tokens", True)
    )["input_ids"]

    # Токены текста
    tokens = input_ids.squeeze(0)

    # Лимит модели по количеству токенов
    max_token_count = getattr(model.config, "max_position_embeddings", 512)

    # Разбиваем текст на чанки
    prefix_length = len(prefix_ids)
    final_prefix_length = len(final_prefix_ids)

    chunk_size = max_token_count - max(prefix_length, final_prefix_length)
    chunks = split_tokens_into_chunks(tokens=tokens, chunk_size=chunk_size, overlap=params.get("overlap", 0))

    # Суммаризуем каждый чанк
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        all_output_ids = []
        for chunk_ids in tqdm(chunks, desc="Summarizing chunks"):
            max_new_tokens = max(int(len(chunk_ids) * 0.30), 200)
            min_length = max(int(len(chunk_ids) * 0.15), 50)

            if len(chunks) == 1 and final_prefix_length > 0:
                chunk_ids = torch.cat((final_prefix_ids, chunk_ids), dim=0).unsqueeze(0).to(device)
            elif prefix_length > 0:
                chunk_ids = torch.cat((prefix_ids, chunk_ids), dim=0).unsqueeze(0).to(device)
            else:
                chunk_ids = chunk_ids.unsqueeze(0).to(device)

            output_ids = model.generate(
                eos_token_id=tokenizer.eos_token_id,
                input_ids=chunk_ids,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                no_repeat_ngram_size=params.get("no_repeat_ngram_size", 2),
                num_beams=params.get("num_beams", 4),
                do_sample=params.get("do_sample", False),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.9),
            )[0]

            all_output_ids.append(output_ids)

    # Декодируем все чанки за раз
    decoded_chunks = tokenizer.batch_decode(all_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    if len(decoded_chunks) > 1:
        new_article = " ".join(decoded_chunks)
        summary = summarize_text(article=new_article, model_name=model_name, tokenizer=tokenizer, model=model, prefix_ids=prefix_ids, final_prefix_ids=final_prefix_ids, params=params)
    else:
        summary = decoded_chunks[0]

    return summary
