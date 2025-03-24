import torch

def summarize_with_rugpt3(article, model_name, tokenizer, model, params):
    # Настройка параметров
    max_length = params.get("max_length", 512)
    add_special_tokens = params.get("add_special_tokens", False)
    no_repeat_ngram_size = params.get("no_repeat_ngram_size", 2)
    padding = params.get("padding", "max_length")
    max_new_tokens = params.get("max_new_tokens", 200)

    # Подготовка текста
    text_tokens = tokenizer(
        article,
        max_length=max_length,
        add_special_tokens=add_special_tokens, 
        padding=padding,
        truncation=True
    )["input_ids"]

    input_ids = text_tokens + [tokenizer.sep_token_id]
    input_ids = torch.LongTensor([input_ids])

    # Генерация текста
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size
    )

    # Обработка результата
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)
    summary = summary.split(tokenizer.sep_token)[1]
    summary = summary.split(tokenizer.eos_token)[0]
    return summary

def summarize_with_fred(article, model_name, tokenizer, model, params):
    # Настройка параметров
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    min_length = params.get("min_length", 30)
    no_repeat_ngram_size = params.get("no_repeat_ngram_size", 2)
    num_beams = params.get("num_beams", 5)
    do_sample = params.get("do_sample", True)
    top_p = params.get("top_p", 0.9)
    max_new_tokens = params.get("max_new_tokens", 200)
    prefix = params.get("prefix", "<LM> Сократи текст.\n")

    # Настройка модели
    model = model.to(device)
    model.eval()

    # Подготовка текста
    src_text = prefix + article
    input_ids=torch.tensor([tokenizer.encode(src_text)]).to(device)

    # Генерация текста
    outputs=model.generate(
        input_ids,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=num_beams,
        min_new_tokens=min_length,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_p=top_p)

    # Обработка результата
    summary = tokenizer.decode(outputs[0][1:])
    return summary

def summarize_text(article, model_name, tokenizer, model, params):
    # Настройка параметров
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    max_length = params.get("max_length", 512)
    min_length = params.get("min_length", 30)
    no_repeat_ngram_size = params.get("no_repeat_ngram_size", 2)
    num_beams = params.get("num_beams", 5)
    add_special_tokens = params.get("add_special_tokens", False)
    do_sample = params.get("do_sample", False)
    top_p = params.get("top_p", 0.9)
    max_new_tokens = params.get("max_new_tokens", 200)
    prefix = params.get("prefix", "Сократи текст.\n")
    padding = params.get("padding", "max_length")

    # Подготовка текста
    src_text = prefix + article
    input_ids = tokenizer(
        src_text,
        return_tensors="pt",
        max_length=max_length,
        padding=padding,
        truncation=True,
        add_special_tokens=add_special_tokens
    )["input_ids"]

    input_ids = input_ids.to(device)

    # Настройка модели
    model = model.to(device)
    model.eval()

    # Генерация текста
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        do_sample=do_sample,
    )[0]

    # Обработка результата
    summary = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return summary