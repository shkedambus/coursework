import time
from typing import List

import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from tqdm import tqdm

from config import model_info
from data import initialize_csv_dataset, initialize_dataset
from qdrant import classify_prediction, embedd_chunks, get_chunks, update_db
from utils import compute_prefix_ids, preprocess_text

# Метрики для оценки моделей
rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")
bertscore = evaluate.load("bertscore")

# Инициализация датасета
NUMBER_OF_TESTS = 1
DATASET_NAME = "csebuetnlp/xlsum"
LANGUAGE = "russian"
FILE_PATH = "lda_train.csv"

# dataset = initialize_dataset(dataset_name=dataset_name, number_of_tests=number_of_tests, language=language)
dataset = initialize_csv_dataset(file_path=FILE_PATH, number_of_tests=NUMBER_OF_TESTS)

def evaluate_single_model(model_name: str, model_info: dict, dataset: Dataset) -> dict:
    """
    Выполняет оценку модели.

    Выполняет суммаризацию для текстов из датасета с помощью модели.
    Подсчитывает значения метрик для результатов суммаризации.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"Evaluating model: {model_name}")
    summarize_function = model_info[model_name]["function"]
    tokenizer = model_info[model_name]["tokenizer"]
    model = model_info[model_name]["model"]
    params = model_info[model_name]["params"]

    # Инициализация модели
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Префикс для суммаризации (e.g. "Сделай краткое изложение:")
    prefix_ids = compute_prefix_ids(tokenizer=tokenizer, prefix=params.get("prefix", ""))

    start = time.time()
    predictions_dataset = dataset.map(
        lambda sample: {
            "prediction": summarize_function(sample["article"], model_name, tokenizer, model, prefix_ids, params)
        },
        batched=False
    )
    elapsed = time.time() - start

    model = model.to("cpu")

    predictions = [sample["prediction"] for sample in predictions_dataset]
    references = [[sample["reference"]] for sample in predictions_dataset]

    # Метрики на уровне всех текстов
    rouge_result = rouge.compute(predictions=predictions, references=references, tokenizer=preprocess_text)
    bleu_result = bleu.compute(predictions=predictions, references=references, tokenize="13a")
    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="ru")

    # Метрики на уровне каждого текста
    per_text_metrics = [
            {
                "prediction": pred,
                "reference": ref[0],
                "rouge": rouge.compute(predictions=[pred], references=[ref], tokenizer=preprocess_text),
                "bleu": bleu.compute(predictions=[pred], references=[ref], tokenize="13a"),
                "bertscore": bertscore.compute(predictions=[pred], references=[ref], lang="ru")
            }
            for pred, ref in zip(predictions, references)
    ]

    result = {
        "ROUGE": rouge_result,
        "BLEU": bleu_result,
        "BERTSCORE": bertscore_result,
        "TIME": elapsed,
        "PER_TEXT_METRICS": per_text_metrics
    }

    return result

def evaluate_models(model_info: dict, dataset: Dataset) -> dict:
    """
    Выполняет оценку для множества моделей,
    запуская функцию оценки для каждой модели
    """
    results = {}
    for model_name, info in model_info.items():
        try:
            results[model_name] = evaluate_single_model(model_name, info, dataset)
        except Exception as e:
            print(f"Error in {model_name}: {e}")
    return results

def evaluate_params(model_info: dict, dataset: Dataset, model_name: str, param_grid: dict) -> List[dict]:
    """
    Выполняет оценку модели на наборе различных параметров.

    Выполняет суммаризацию для текстов из датасета с помощью модели для каждого набора параметров.
    Подсчитывает значения метрик для результатов суммаризации для каждого набора параметров.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"Evaluating model: {model_name}")
    summarize_function = model_info[model_name]["function"]
    tokenizer = model_info[model_name]["tokenizer"]
    model = model_info[model_name]["model"]
    params = model_info[model_name]["params"]

    # Инициализация модели один раз
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Префикс для суммаризации (e.g. "Сделай краткое изложение:")
    prefix_ids = compute_prefix_ids(tokenizer=tokenizer, prefix=params.get("prefix", ""))
    
    results = []
    for temp in tqdm(param_grid["temperature"], desc="Temperature"):
        for top_p in tqdm(param_grid["top_p"], desc="Top_p"):
            for overlap in tqdm(param_grid["overlapping"], desc="Overlap"):

                # Применяем параметры к модели
                params.update({"do_sample": True, "temperature": temp, "top_p": top_p, "overlap": overlap})

                try:
                    start = time.time()
                    predictions_dataset = dataset.map(
                        lambda sample: {
                            "prediction": summarize_function(sample["article"], model_name, tokenizer, model, prefix_ids, params)
                        },
                        batched=False
                    )
                    elapsed = time.time() - start

                    predictions = [sample["prediction"] for sample in predictions_dataset]
                    references = [[sample["reference"]] for sample in predictions_dataset]

                    # Метрики на уровне всех текстов
                    rouge_result = rouge.compute(predictions=predictions, references=references, tokenizer=preprocess_text)
                    bleu_result = bleu.compute(predictions=predictions, references=references, tokenize="13a")
                    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="ru")

                    # Метрики на уровне каждого текста
                    per_text_metrics = [
                            {
                                "prediction": pred,
                                "reference": ref[0],
                                "rouge": rouge.compute(predictions=[pred], references=[ref], tokenizer=preprocess_text),
                                "bleu": bleu.compute(predictions=[pred], references=[ref], tokenize="13a"),
                                "bertscore": bertscore.compute(predictions=[pred], references=[ref], lang="ru")
                            }
                            for pred, ref in zip(predictions, references)
                    ]

                    results.append({
                        "temperature": temp,
                        "top_p": top_p,
                        "overlap": overlap,
                        "ROUGE": rouge_result,
                        "BLEU": bleu_result,
                        "BERTSCORE": bertscore_result,
                        "PER_TEXT_METRICS": per_text_metrics,
                        "TIME": elapsed
                    })

                except Exception as e:
                    print(f"Error (temperature={temp}, top_p={top_p}, overlapping={overlap}): {e}")
                    continue

    return results

def build_table(results, kind: str = "model") -> pd.DataFrame:
    """
    Строит таблицу по результатам оценки и сохраняет её в .csv файл.
    """
    rows = []

    def avg(metric):
        return sum(metric) / len(metric)

    for item in results.values() if kind == "model" else results:
        base = {
            "rouge-1": round(item["ROUGE"]["rouge1"], 3),
            "rouge-2": round(item["ROUGE"]["rouge2"], 3),
            "rouge-L": round(item["ROUGE"]["rougeL"], 3),
            "rouge-Lsum": round(item["ROUGE"]["rougeLsum"], 3),
            "bleu-score": round(item["BLEU"]["score"], 3),
            "bertscore-precision": round(avg(item["BERTSCORE"]["precision"]), 3),
            "bertscore-recall": round(avg(item["BERTSCORE"]["recall"]), 3),
            "bertscore-f1": round(avg(item["BERTSCORE"]["f1"]), 3),
            "time (seconds)": round(item["TIME"], 3),
        }
        if kind == "model":
            base["model"] = list(results.keys())[list(results.values()).index(item)]
        else:
            base.update({k: item[k] for k in ("temperature", "top_p", "overlap")})
        rows.append(base)

    df = pd.DataFrame(rows)
    file = f"results_{kind}.csv"
    df.to_csv(file, index=False)
    print(f"\nТаблица сохранена в файл: {file}")
    return df

def visualize(df: pd.DataFrame, metrics: List[str], palettes: List[str]) -> None:
    """
    Визуализирует результаты оценки модели на наборе различных параметров.

    Использует библиотеки seaborn и matplotlib для построения heatmap'ы для каждой метрики.
    (Результаты усредняются по параметру overlap).
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=True)
    for i, (metric, palette) in enumerate(zip(metrics, palettes)):
        pivot = df.pivot_table(index="temperature", columns="top_p", values=metric, aggfunc="mean")
        sns.heatmap(pivot, ax=axes[i], annot=True, fmt=".2f", cmap=palette, cbar_kws={"label": metric})
        axes[i].set_title(metric.upper())
        axes[i].set_xlabel("top_p")
        if i == 0:
            axes[i].set_ylabel("temperature")
    plt.tight_layout()
    plt.show()

# df = pd.read_csv("params_result.csv")
# metrics = ["rouge-1", "rouge-L", "bleu-score", "bertscore-f1"]
# palettes = ["YlGn_r", "Blues_r", "Oranges_r", "Purples_r"]
# visualize(df=df, metrics=metrics, palettes=palettes)

def collect_bad_results(results: dict) -> List:
    """
    Собирает 'плохие' на основе полученных метрик суммаризации моделей в один большой список.
    """
    data = []
    for model_name, metrics in results.items():
        per_text_metrics = metrics["PER_TEXT_METRICS"]
        for idx, ptm in enumerate(per_text_metrics):
            prediction = ptm["prediction"]

            rouge1 = round(float(ptm["rouge"]["rouge1"]), 3)
            rougeL = round(float(ptm["rouge"]["rougeL"]), 3)
            bleu = round(float(ptm["bleu"]["score"]), 3)
            bertscore_f1 = round(float(ptm["bertscore"]["f1"][0]), 3)

            if rouge1 <= 0.5 or rougeL <= 0.5 or bleu <= 5 or bertscore_f1 <= 0.55:
                data.append(prediction)

        return data
    
def upsert_good_predictions(dataset: Dataset) -> None:
    """
    Загружает эталонные суммаризации в базу данных.
    (В коллекцию 'good_predictions').
    """
    good_predictions = [sample["reference"] for sample in dataset]
    embeddings = embedd_chunks(good_predictions)
    update_db("good_predictions", good_predictions, embeddings)

def upsert_bad_predictions(bad_predictions: List) -> None:
    """
    Загружает 'плохие' (с галлюцинациями) суммаризации в базу данных.
    (В коллекцию 'bad_predictions').
    """
    embeddings = embedd_chunks(bad_predictions)
    update_db("bad_predictions", bad_predictions, embeddings)

def test_model(model_name: str, model_info: dict, dataset: Dataset) -> dict:
    """
    Оценивает работу модели на заданном датасете и классифицирует каждое предсказание как 'хорошее' или 'плохое' 
    на основе сходства с эмбеддингами, хранящимися в векторной базе данных.
    """
    results = evaluate_single_model(model_name=model_name, model_info=model_info, dataset=dataset)
    per_text_metrics = results[model_name]["PER_TEXT_METRICS"]

    scores = {}
    for idx, metric in enumerate(per_text_metrics):
        prediction = metric["prediction"]
        good_prediction = get_chunks("good_predictions", prediction)
        bad_prediction = get_chunks("bad_predictions", prediction)
        
        scores[idx] = classify_prediction(prediction, good_prediction, bad_prediction)

    return scores
