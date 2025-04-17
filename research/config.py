from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer, GPT2Tokenizer,
                          MBartForConditionalGeneration, MBartTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer)

from generator.summarize import summarize_text, summarize_with_rugpt3

base_params = {
    "device": "cuda",
    "no_repeat_ngram_size": 2,
    "num_beams": 4,
}

params_prefix = {
    **base_params,
    "prefix": "<LM> Сделай краткое изложение:\n",
}

params_no_prefix = {
    **base_params,
    "prefix": "",
}

params_fred = {
    **base_params,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.7,
    "overlap": 0.2,
    "prefix": "",
}

model_info = {
    "facebook/bart-large-cnn": {
        "function": summarize_text,
        "tokenizer": AutoTokenizer.from_pretrained("facebook/bart-large-cnn"),
        "model": AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn"),
        "params": params_prefix
    },
    "RussianNLP/FRED-T5-Summarizer": {
        "function": summarize_text,
        "tokenizer": GPT2Tokenizer.from_pretrained("RussianNLP/FRED-T5-Summarizer", eos_token="</s>"),
        "model": T5ForConditionalGeneration.from_pretrained("RussianNLP/FRED-T5-Summarizer"),
        "params": params_fred
    },
    "IlyaGusev/rugpt3medium_sum_gazeta": {
        "function": summarize_with_rugpt3,
        "tokenizer": AutoTokenizer.from_pretrained("IlyaGusev/rugpt3medium_sum_gazeta"),
        "model": AutoModelForCausalLM.from_pretrained("IlyaGusev/rugpt3medium_sum_gazeta"),
        "params": params_no_prefix
    },
    "Falconsai/text_summarization": {
        "function": summarize_text,
        "tokenizer": AutoTokenizer.from_pretrained("Falconsai/text_summarization"),
        "model": AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization"),
        "params": params_no_prefix
    },
    "sshleifer/distilbart-cnn-12-6": {
        "function": summarize_text,
        "tokenizer": AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6"),
        "model": AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6"),
        "params": params_prefix
    },
    "sansmislom/mt5-small-finetuned-gazeta-ru": {
        "function": summarize_text,
        "tokenizer": AutoTokenizer.from_pretrained("sansmislom/mt5-small-finetuned-gazeta-ru", legacy=False, use_fast=False),
        "model": AutoModelForSeq2SeqLM.from_pretrained("sansmislom/mt5-small-finetuned-gazeta-ru"),
        "params": params_prefix
    },
    "csebuetnlp/mT5_multilingual_XLSum": {
        "function": summarize_text,
        "tokenizer": AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum", legacy=False, use_fast=False),
        "model": AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum"),
        "params": params_prefix
    },
    "IlyaGusev/mbart_ru_sum_gazeta": {
        "function": summarize_text,
        "tokenizer": MBartTokenizer.from_pretrained("IlyaGusev/mbart_ru_sum_gazeta"),
        "model": MBartForConditionalGeneration.from_pretrained("IlyaGusev/mbart_ru_sum_gazeta"),
        "params": params_prefix
    },
    "IlyaGusev/rut5_base_sum_gazeta": {
        "function": summarize_text,
        "tokenizer": AutoTokenizer.from_pretrained("IlyaGusev/rut5_base_sum_gazeta"),
        "model": T5ForConditionalGeneration.from_pretrained("IlyaGusev/rut5_base_sum_gazeta"),
        "params": params_prefix
    },
    "utrobinmv/t5_summary_en_ru_zh_large_2048": {
        "function": summarize_text,
        "tokenizer": T5Tokenizer.from_pretrained("utrobinmv/t5_summary_en_ru_zh_large_2048"),
        "model": T5ForConditionalGeneration.from_pretrained("utrobinmv/t5_summary_en_ru_zh_large_2048"),
        "params": params_prefix
    },
    "utrobinmv/t5_summary_en_ru_zh_base_2048": {
        "function": summarize_text,
        "tokenizer": T5Tokenizer.from_pretrained("utrobinmv/t5_summary_en_ru_zh_base_2048"),
        "model": T5ForConditionalGeneration.from_pretrained("utrobinmv/t5_summary_en_ru_zh_base_2048"),
        "params": params_prefix
    },
    "Nehc/mT5_ru_XLSum": {
        "function": summarize_text,
        "tokenizer": AutoTokenizer.from_pretrained("Nehc/mT5_ru_XLSum"),
        "model": AutoModelForSeq2SeqLM.from_pretrained("Nehc/mT5_ru_XLSum"),
        "params": params_prefix
    },
    "sacreemure/med_t5_summ_ru": {
        "function": summarize_text,
        "tokenizer": AutoTokenizer.from_pretrained("sacreemure/med_t5_summ_ru"),
        "model": AutoModelForSeq2SeqLM.from_pretrained("sacreemure/med_t5_summ_ru"),
        "params": params_prefix
    },
    "sarahai/ru-sum": {
        "function": summarize_text,
        "tokenizer": AutoTokenizer.from_pretrained("sarahai/ru-sum"),
        "model": AutoModelForSeq2SeqLM.from_pretrained("sarahai/ru-sum"),
        "params": params_prefix
    },
    "sarahai/ruT5-base-summarizer": {
        "function": summarize_text,
        "tokenizer": T5Tokenizer.from_pretrained("sarahai/ruT5-base-summarizer"),
        "model": T5ForConditionalGeneration.from_pretrained("sarahai/ruT5-base-summarizer"),
        "params": params_prefix
    },
    "csebuetnlp/mT5_m2o_russian_crossSum": {
        "function": summarize_text,
        "tokenizer": AutoTokenizer.from_pretrained("csebuetnlp/mT5_m2o_russian_crossSum", legacy=False, use_fast=False),
        "model": AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_m2o_russian_crossSum"),
        "params": params_prefix
    },
}
