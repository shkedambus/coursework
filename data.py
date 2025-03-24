import pandas as pd
from datasets import load_dataset, Dataset

def initialize_dataset(dataset_name, number_of_tests, language="russian"):
    dataset = load_dataset(dataset_name, language)
    test_data = dataset["test"]

    filtered_articles = []
    filtered_references = []
    count = 0

    # Идем по всему датасету, пока не соберем нужное количество текстов
    for sample in test_data:
        article = sample["text"].strip()
        reference = sample["summary"].strip()

        if article and len(article.split()) <= 512:
            filtered_articles.append(article)
            filtered_references.append(reference)
            count += 1

        if count == number_of_tests:
            break

    # Создаем Dataset
    filtered_dataset = Dataset.from_dict({
        "article": filtered_articles,
        "reference": filtered_references
    })

    return filtered_dataset

def initialize_csv_dataset(file_path, number_of_tests):
    df = pd.read_csv(file_path, delimiter=";")

    filtered_df = df.dropna(subset=["cleaned", "summarization"])

    filtered_df = filtered_df.head(number_of_tests)

    # Преобразуем в Hugging Face Dataset
    dataset = Dataset.from_pandas(filtered_df[["cleaned", "summarization"]])
    dataset = dataset.rename_columns({"cleaned": "article", "summarization": "reference"})

    return dataset