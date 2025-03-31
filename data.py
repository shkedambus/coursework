import pandas as pd
from datasets import Dataset, load_dataset


def initialize_dataset(dataset_name: str, number_of_tests: int, language="russian") -> Dataset:
    """
    Инициализирует датасет с помощью библиотеки datasets.
    """
    dataset = load_dataset(dataset_name, language)
    test_data = dataset["test"]

    selected_data = test_data.select(range(number_of_tests))

    filtered_dataset = Dataset.from_dict({
        "article": [sample["text"].strip() for sample in selected_data],
        "reference": [sample["summary"].strip() for sample in selected_data]
    })

    return filtered_dataset

def initialize_csv_dataset(file_path: str, number_of_tests: int) -> Dataset:
    """
    Инициализирует датасет из файла .csv.
    """
    df = pd.read_csv(file_path, delimiter=";")

    filtered_df = df.dropna(subset=["cleaned", "summarization"])
    filtered_df = filtered_df.head(number_of_tests)

    # Преобразуем в Hugging Face Dataset
    dataset = Dataset.from_pandas(filtered_df[["cleaned", "summarization"]])
    dataset = dataset.rename_columns({"cleaned": "article", "summarization": "reference"})

    return dataset
