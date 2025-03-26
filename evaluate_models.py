import torch
from tqdm import tqdm
import evaluate_models
import pandas as pd
import time
from random import randint

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2Tokenizer
from transformers import MBartTokenizer, MBartForConditionalGeneration

from summarize import summarize_text, summarize_with_rugpt3
from data import initialize_dataset, initialize_csv_dataset

from razdel import tokenize

from qdrant import get_chunks, embedd_chunks, update_db, classify_prediction

# Функция для токенизации текста на русском языке (для метрики ROUGE)
def preprocess_text(text):
    tokens = [token.text for token in tokenize(text)]
    return " ".join(tokens)

params_prefix = {
    "device": "cuda",
    "no_repeat_ngram_size": 2,
    "num_beams": 4,
    "prefix": "<LM> Сделай краткое изложение:\n",
}

params_no_prefix = {
    "device": "cuda",
    "no_repeat_ngram_size": 2,
    "num_beams": 4,
}

model_data = {
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
        "params": params_no_prefix
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

NUMBER_OF_TESTS = 1
DATASET_NAME = "csebuetnlp/xlsum"
LANGUAGE = "russian"
FILE_PATH = "lda_train.csv"

# dataset = initialize_dataset(dataset_name=DATASET_NAME, number_of_tests=NUMBER_OF_TESTS, language=LANGUAGE)
dataset = initialize_csv_dataset(file_path=FILE_PATH, number_of_tests=NUMBER_OF_TESTS)

def evaluate_models(dataset):
    rouge = evaluate_models.load("rouge")
    bleu = evaluate_models.load("sacrebleu")
    bertscore = evaluate_models.load("bertscore")
    results = {}

    for model_name in model_data:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print(f"Evaluating model: {model_name}")
        summarize_function = model_data[model_name]["function"]
        tokenizer = model_data[model_name]["tokenizer"]
        model = model_data[model_name]["model"]
        params = model_data[model_name]["params"]

        try:

            # Инициализация модели один раз
            device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

            # Префикс для суммаризации (e.g. "Сделай краткое изложение:")
            prefix = params.get("prefix", "")
            if prefix.strip():
                prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            else:
                prefix_ids = torch.tensor([], dtype=torch.long)

            start_time = time.time()

            predictions_dataset = dataset.map(
                lambda sample: {
                    "prediction": summarize_function(sample["article"], model_name, tokenizer, model, prefix_ids, params)
                },
                batched=False
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Освобождаем видеопамять
            model = model.to("cpu")

            predictions_texts = [sample["prediction"] for sample in predictions_dataset]
            references = [[sample["reference"]] for sample in predictions_dataset]

            # Метрики на уровне всех текстов
            rouge_result = rouge.compute(predictions=predictions_texts, references=references, tokenizer=preprocess_text)
            bleu_result = bleu.compute(predictions=predictions_texts, references=references, tokenize="13a")
            bertscore_result = bertscore.compute(predictions=predictions_texts, references=references, lang="ru")

            # Метрики на уровне каждого текста
            per_text_metrics = []
            for i, prediction in enumerate(predictions_texts):
                ref = references[i]
                rouge_per_text = rouge.compute(predictions=[prediction], references=[ref], tokenizer=preprocess_text)
                bleu_per_text = bleu.compute(predictions=[prediction], references=[ref], tokenize="13a")
                bertscore_per_text = bertscore.compute(predictions=[prediction], references=[ref], lang="ru")
                
                per_text_metrics.append({
                    "prediction": prediction,
                    "reference": ref[0],
                    "rouge": rouge_per_text,
                    "bleu": bleu_per_text,
                    "bertscore": bertscore_per_text
                })

            # Сохраняем результаты
            results[model_name] = {
                "ROUGE": rouge_result,
                "BLEU": bleu_result,
                "BERTSCORE": bertscore_result,
                "TIME": elapsed_time,
                "PER_TEXT_METRICS": per_text_metrics
            }

        except Exception as e:
            print(f"Error in {model_name}: {e}")
            continue

    return results

# def evaluate_params(dataset, model_name, params_grid):
#     rouge = evaluate.load("rouge")
#     bleu = evaluate.load("sacrebleu")
#     bertscore = evaluate.load("bertscore")
#     results = []

#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()

#     print(f"Evaluating model: {model_name}")
#     summarize_function = model_data[model_name]["function"]
#     tokenizer = model_data[model_name]["tokenizer"]
#     model = model_data[model_name]["model"]
#     params = model_data[model_name]["params"]

#     # Инициализация модели один раз
#     device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()

#     # Префикс для суммаризации (e.g. "Сделай краткое изложение:")
#     prefix = params.get("prefix", "")
#     if prefix.strip():
#         prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
#     else:
#         prefix_ids = torch.tensor([], dtype=torch.long)
    
#     for temp in tqdm(params_grid["temperature"], desc="Temperature"):
#         for top_p in tqdm(params_grid["top_p"], desc="Top_p"):
#             for overlap in tqdm(params_grid["overlapping"], desc="Overlap"):
#                 # Применяем параметры к модели
#                 params["do_sampling"] = True
#                 params["temperature"] = temp
#                 params["top_p"] = top_p
#                 params["overlapping"] = overlap

#                 start_time = time.time()

#                 predictions_dataset = dataset.map(
#                     lambda sample: {
#                         "prediction": summarize_function(sample["article"], model_name, tokenizer, model, prefix_ids, params)
#                     },
#                     batched=False
#                 )

#                 end_time = time.time()
#                 elapsed_time = end_time - start_time

#                 predictions_texts = [sample["prediction"] for sample in predictions_dataset]
#                 references = [[sample["reference"]] for sample in predictions_dataset]

#                 # Метрики на уровне всех текстов
#                 rouge_result = rouge.compute(predictions=predictions_texts, references=references, tokenizer=preprocess_text)
#                 bleu_result = bleu.compute(predictions=predictions_texts, references=references, tokenize="13a")
#                 bertscore_result = bertscore.compute(predictions=predictions_texts, references=references, lang="ru")

#                 # Метрики на уровне каждого текста
#                 per_text_metrics = []
#                 for i, prediction in enumerate(predictions_texts):
#                     ref = references[i]
#                     rouge_per_text = rouge.compute(predictions=[prediction], references=[ref], tokenizer=preprocess_text)
#                     bleu_per_text = bleu.compute(predictions=[prediction], references=[ref], tokenize="13a")
#                     bertscore_per_text = bertscore.compute(predictions=[prediction], references=[ref], lang="ru")
                    
#                     per_text_metrics.append({
#                         "prediction": prediction,
#                         "reference": ref[0],
#                         "rouge": rouge_per_text,
#                         "bleu": bleu_per_text,
#                         "bertscore": bertscore_per_text
#                     })

#                 results.append({
#                     "temperature": temp,
#                     "top_p": top_p,
#                     "overlap": overlap,
#                     "ROUGE": rouge_result,
#                     "BLEU": bleu_result,
#                     "BERTSCORE": bertscore_result,
#                     "PER_TEXT_METRICS": per_text_metrics,
#                     "TIME": elapsed_time
#                 })
    
#     return results

def build_table(results, type):
    output = []
    if type == "model":
        for model_name, metrics in results.items():
            bertscore_precision = sum(metrics["BERTSCORE"]["precision"]) / len(metrics["BERTSCORE"]["precision"])
            bertscore_recall = sum(metrics["BERTSCORE"]["recall"]) / len(metrics["BERTSCORE"]["recall"])
            bertscore_f1 = sum(metrics["BERTSCORE"]["f1"]) / len(metrics["BERTSCORE"]["f1"])
            elapsed_time = metrics["TIME"]

            output.append({
                "model": model_name,
                "rouge-1": round(float(metrics["ROUGE"]["rouge1"]), 3),
                "rouge-2": round(float(metrics["ROUGE"]["rouge2"]), 3),
                "rouge-L": round(float(metrics["ROUGE"]["rougeL"]), 3),
                "rouge-Lsum": round(float(metrics["ROUGE"]["rougeLsum"]), 3),
                "bleu-score": round(metrics["BLEU"]["score"], 3),
                "bertscore-precision": round(bertscore_precision, 3),
                "bertscore-recall": round(bertscore_recall, 3),
                "bertscore-f1": round(bertscore_f1, 3),
                "time (seconds)": round(elapsed_time, 3)
            })

        df = pd.DataFrame(output)

        id = str(randint(0, 100))
        file_name = f"results_table_{id}.csv"

        df.to_csv(file_name, index=False, encoding="utf-8")
        print(f"\nТаблица результатов сохранена в файл: {file_name}")
    
    # elif type == "params":
    #     for result in results:
    #         bertscore_precision = sum(result["BERTSCORE"]["precision"]) / len(result["BERTSCORE"]["precision"])
    #         bertscore_recall = sum(result["BERTSCORE"]["recall"]) / len(result["BERTSCORE"]["recall"])
    #         bertscore_f1 = sum(result["BERTSCORE"]["f1"]) / len(result["BERTSCORE"]["f1"])
    #         elapsed_time = result["TIME"]

    #         output.append({
    #             "rouge-1": round(float(result["ROUGE"]["rouge1"]), 3),
    #             "rouge-2": round(float(result["ROUGE"]["rouge2"]), 3),
    #             "rouge-L": round(float(result["ROUGE"]["rougeL"]), 3),
    #             "rouge-Lsum": round(float(result["ROUGE"]["rougeLsum"]), 3),
    #             "bleu-score": round(result["BLEU"]["score"], 3),
    #             "bertscore-precision": round(bertscore_precision, 3),
    #             "bertscore-recall": round(bertscore_recall, 3),
    #             "bertscore-f1": round(bertscore_f1, 3),
    #             "time (seconds)": round(elapsed_time, 3)
    #         })

    #     df = pd.DataFrame(output)
    #     file_name = f"{model_name}.csv"

    #     df.to_csv(file_name, index=False, encoding="utf-8")
    #     print(f"\nТаблица результатов сохранена в файл: {file_name}")

    return df

def collect_bad_results(results):
    data = []
    for model_name, metrics in results.items():
        per_text_metrics = metrics["PER_TEXT_METRICS"]
        for idx, ptm in enumerate(per_text_metrics):
            prediction = ptm["prediction"]
            reference = ptm["reference"]

            rouge1 = round(float(ptm["rouge"]["rouge1"]), 3)
            rouge2 = round(float(ptm["rouge"]["rouge2"]), 3)
            rougeL = round(float(ptm["rouge"]["rougeL"]), 3)
            rougeLsum = round(float(ptm["rouge"]["rougeLsum"]), 3)
            bleu = round(float(ptm["bleu"]["score"]), 3)
            bertscore_precision = round(float(ptm["bertscore"]["precision"][0]), 3)
            bertscore_recall = round(float(ptm["bertscore"]["recall"][0]), 3)
            bertscore_f1 = round(float(ptm["bertscore"]["f1"][0]), 3)

            output = []
            output.append({
                "rouge-1": rouge1,
                "rouge-2": rouge2,
                "rouge-L": rougeL,
                "rouge-Lsum": rougeLsum,
                "bleu-score": bleu,
                "bertscore-precision": bertscore_precision,
                "bertscore-recall": bertscore_recall,
                "bertscore-f1": bertscore_f1,
            })

            df = pd.DataFrame(output)

            print(f"Model: {model_name}")
            print(f"Reference:\n{reference}")
            print(f"Prediction:\n{prediction}")
            print(df.to_string(index=False), end=2*"\n")

            if rouge1 <= 0.4 or bleu <= 5 or bertscore_f1 <= 0.55:
                data.append(prediction)

        return data
    
def upsert_good_predictions():
    good_predictions = [sample["reference"] for sample in dataset]
    embeddings = embedd_chunks(good_predictions)
    update_db("good_predictions", good_predictions, embeddings)

def upsert_bad_predictions():
    results = evaluate_models()
    bad_predictions = collect_bad_results(results)
    embeddings = embedd_chunks(bad_predictions)
    update_db("bad_predictions", bad_predictions, embeddings)

def test_model(model_to_test):
    model_name = list(model_to_test.keys())[0]
    results = evaluate_models(model_to_test)
    per_text_metrics = results[model_name]["PER_TEXT_METRICS"]

    scores = {}
    for idx, metric in enumerate(per_text_metrics):
        prediction = metric["prediction"]
        good_prediction = get_chunks("good_predictions", prediction)
        bad_prediction = get_chunks("bad_predictions", prediction)
        
        scores[idx] = classify_prediction(prediction, good_prediction, bad_prediction)

    return scores