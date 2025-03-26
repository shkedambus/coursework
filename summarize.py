import torch
from tqdm import tqdm

def summarize_with_rugpt3(article, model_name, tokenizer, model, prefix_ids, params):
    # Настройка параметров
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    no_repeat_ngram_size = params.get("no_repeat_ngram_size", 2)
    num_beams = params.get("num_beams", 4)
    add_special_tokens = params.get("add_special_tokens", True)

    input_ids = tokenizer(
        article,
        return_tensors="pt",
        padding=False,
        truncation=False,   
        add_special_tokens=add_special_tokens
    )["input_ids"]

    # Количество токенов в тексте
    tokens = input_ids.squeeze(0)
    token_count = len(tokens)

    # Лимит модели по количеству токенов
    max_token_count = getattr(model.config, "max_position_embeddings", 512) // 2

    chunks = [tokens[i:i + max_token_count] for i in range(0, token_count, max_token_count)]
    chunks = [chunk for chunk in chunks if len(chunk) > 0]

    with torch.inference_mode():
        all_output_ids = []
        for chunk_ids in chunks:
        # for chunk_ids in tqdm(chunks, desc="Summarizing chunks"):
            max_new_tokens = max(int(len(chunk_ids) * 0.35), 200)
            min_length = max(int(len(chunk_ids) * 0.15), 50)

            eos_id = torch.tensor([tokenizer.sep_token_id], dtype=chunk_ids.dtype, device=chunk_ids.device)

            if len(prefix_ids) > 0:
                chunk_ids = torch.cat((prefix_ids, chunk_ids), dim=0)

            chunk_ids = torch.cat((chunk_ids, eos_id), dim=0)

            chunk_ids = chunk_ids.unsqueeze(0).to(device)

            output_ids = model.generate(
                input_ids=chunk_ids,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_beams=num_beams,
                do_sample=False,
            )[0]

            all_output_ids.append(output_ids)

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

def summarize_text(article, model_name, tokenizer, model, prefix_ids, params):
    # Настройка параметров
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    no_repeat_ngram_size = params.get("no_repeat_ngram_size", 2)
    num_beams = params.get("num_beams", 4)
    add_special_tokens = params.get("add_special_tokens", True)
    do_sample = params.get("do_sample", False)
    temperature = params.get("temperature", 0)
    top_p = params.get("top_p", 1)
    overlapping = params.get("overlapping", 0)

    input_ids = tokenizer(
        article,
        return_tensors="pt",
        padding=False,
        truncation=False,   
        add_special_tokens=add_special_tokens
    )["input_ids"]

    # Количество токенов в тексте
    tokens = input_ids.squeeze(0)
    token_count = len(tokens)

    # Лимит модели по количеству токенов
    max_token_count = getattr(model.config, "max_position_embeddings", 512)

    available_chunk_len = max_token_count - len(prefix_ids)
    stride = int(available_chunk_len * (1 - overlapping))

    chunks = [tokens[i:i + available_chunk_len] for i in range(0, token_count, stride)]
    chunks = [chunk for chunk in chunks if len(chunk) > 0]

    with torch.inference_mode():
        all_output_ids = []
        for chunk_ids in chunks:
        # for chunk_ids in tqdm(chunks, desc="Summarizing chunks"):
            max_new_tokens = max(int(len(chunk_ids) * 0.35), 200)
            min_length = max(int(len(chunk_ids) * 0.15), 50)

            if len(prefix_ids) > 0:
                chunk_ids = torch.cat((prefix_ids, chunk_ids), dim=0).unsqueeze(0).to(device)
            else:
                chunk_ids = chunk_ids.unsqueeze(0).to(device)

            output_ids = model.generate(
                input_ids=chunk_ids,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )[0]

            all_output_ids.append(output_ids)

    decoded_chunks = tokenizer.batch_decode(all_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    if len(decoded_chunks) > 1:
        new_article = " ".join(decoded_chunks)
        summary = summarize_text(article=new_article, model_name=model_name, tokenizer=tokenizer, model=model, prefix_ids=prefix_ids, params=params)
    else:
        summary = decoded_chunks[0]

    return summary
