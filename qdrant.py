import hashlib
import uuid
from typing import List

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Эмбеддинг-модель
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
VECTOR_SIZE = 1536
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True, device="cuda")

# Подключаемся к локальному Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
qdrant_client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

from logger import get_logger
logger = get_logger("Qdrant")

def ensure_collections_exist(collections: List[str], vector_size: int) -> None:
    """
    Создает нужные коллекции в базе Qdrant, если их не существует.
    """
    for name in collections:
        if not qdrant_client.collection_exists(name):
            qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

def delete_collection(collection_name: str) -> None:
    """
    Удаляет указанную коллекцию из базы Qdrant, если она существует.
    """
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
        logger.info(f"Коллекция '{collection_name}' удалена.")
    else:
        logger.info(f"Коллекция '{collection_name}' не существует.")

def clear_collection(collection_name: str) -> None:
    """
    Удаляет все объекты в коллекции, оставляя саму коллекцию пустой.
    """
    if not qdrant_client.collection_exists(collection_name):
        logger.info(f"Коллекция '{collection_name}' не существует.")
        return

    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=Filter(must=[])
    )

    logger.info(f"Все данные в коллекции '{collection_name}' удалены.")

def delete_user_data(collection_name: str, user_id: int) -> None:
    """
    Удаляет все данные, связанные с конкретным пользователем, из указанной коллекции.
    """
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector={"filter": {"must": [{"key": "user_id", "match": {"value": user_id}}]}}
        )
        logger.info(f"Все данные пользователя {user_id} удалены из коллекции '{collection_name}'.")
    else:
        logger.info(f"Коллекция '{collection_name}' не существует.")

def inspect_collection(collection_name: str, limit: int = 20) -> None:
    """
    Выводит информацию о содержимом коллекции - ID пользователей и связанные с ними тексты.
    """
    if not qdrant_client.collection_exists(collection_name):
        logger.info(f"Коллекция '{collection_name}' не существует.")
        return

    points = qdrant_client.scroll(
        collection_name=collection_name,
        limit=limit
    )[0]

    if not points:
        logger.info(f"Коллекция '{collection_name}' пуста.")
        return
    
    logger.info(f"Данные в коллекции '{collection_name}':")
    for point in points:
        logger.info(f"ID: {point.id}, User ID: {point.payload.get('user_id', 'N/A')}")
        logger.info(f"Текст: {point.payload.get('text', 'Нет текста')}")
        logger.info("-" * 50)

def embedd_chunks(chunks: List[str]) -> torch.Tensor:
    """
    Строит эмбеддинги для уникальных текстовых чанков с помощью модели Qwen и возвращает тензор.
    """
    chunks = list(set(chunks))

    embeddings = embedding_model.encode(
        chunks,
        batch_size=32,
        convert_to_tensor=True,    # Ускоряет на GPU
        normalize_embeddings=True  # Ускоряет поиск по cosine similarity
    )

    return embeddings

# Функция для создания уникального хэша для текста чанка
def hash_text(text: str) -> str:
    """
    Создает уникальный хэш для текста чанка с использованием алгоритма SHA-256.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Функция для проверки, есть ли уже такой хэш в базе
def is_duplicate_chunk(collection_name: str, chunk_hash: str) -> bool:
    """
    Проверяет, существует ли в коллекции чанк с таким же хэшем.
    """
    existing_chunks, _ = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter={"must": [{"key": "chunk_hash", "match": {"value": chunk_hash}}]},
        limit=1
    )
    return len(existing_chunks) > 0

def update_db(collection_name: str, chunks: List[str], embeddings: torch.Tensor, user_id: int) -> None:
    """
    Добавляет уникальные чанки и их эмбеддинги в коллекцию Qdrant, исключая дубликаты.
    """
    unique_chunks = []
    for chunk, emb in zip(chunks, embeddings):
        chunk_hash = hash_text(chunk)

        if is_duplicate_chunk(collection_name, chunk_hash):
            continue

        point_id = int(uuid.uuid4().int % 1e9)

        unique_chunks.append(
            PointStruct(
                id=point_id,
                vector=emb,
                payload={"text": chunk, "user_id": user_id, "chunk_hash": chunk_hash}
            )
        )

    if unique_chunks:
        qdrant_client.upsert(collection_name=collection_name, points=unique_chunks)
        logger.info(f"Добавлено {len(unique_chunks)} новых чанков для пользователя {user_id}")
    else:
        logger.info("Все чанки уже есть в базе, ничего не добавлено.")

def get_chunks(collection_name: str, query: str, user_id: int = 0) -> List[str]:
    """
    Находит наиболее релевантные чанки из коллекции по эмбеддингу запроса.
    """
    query_vector = embedding_model.encode([query])[0]
    retrieved_chunks = similarity_search(collection_name, query_vector, user_id)
    return retrieved_chunks

def similarity_search(collection_name: str, query_vector: torch.Tensor, user_id: int, threshold: float = 0.3, top_k: int = 10) -> List[str]:
    """
    Выполняет поиск ближайших чанков в коллекции по косинусному сходству и возвращает тексты.
    """
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=threshold,
        query_filter={"must": [{"key": "user_id", "match": {"value": user_id}}]}
    )

    retrieved_texts = [hit.payload["text"] for hit in search_result]
    return retrieved_texts

def average_score(scores: torch.Tensor) -> float:
    """
    Вычисляет среднее значение из тензора метрик, поддерживает матрицы и вектора.
    """
    if isinstance(scores[0], list):  # Если scores — матрица
        flat_scores = [score for row in scores for score in row]  # Разворачиваем в одномерный список
    else:
        flat_scores = scores

    return round(float(sum(flat_scores) / len(flat_scores)), 3)

def cosine_similarity_pytorch(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет косинусное сходство между двумя тензорами, используя PyTorch.
    """     
    # Нормализуем векторы
    tensor1_normalized = tensor1 / tensor1.norm(dim=1, keepdim=True)
    tensor2_normalized = tensor2 / tensor2.norm(dim=1, keepdim=True)

    # Вычисляем косинусное сходство
    cosine_sim = torch.mm(tensor1_normalized, tensor2_normalized.transpose(0, 1))
    return cosine_sim

def compare_embeddings(predictions: List[str], references: List[str]) -> torch.Tensor:
    """
    Сравнивает эмбеддинги предсказаний и эталонных ответов, возвращая матрицу сходства.
    """
    prediction_embeddings = embedding_model.encode(
        predictions,
        batch_size=32,
        convert_to_tensor=True,    # Ускоряет на GPU
        normalize_embeddings=True  # Ускоряет поиск по cosine similarity
    )

    reference_embeddings = embedding_model.encode(
        references,
        batch_size=32,
        convert_to_tensor=True,    # Ускоряет на GPU
        normalize_embeddings=True  # Ускоряет поиск по cosine similarity
    )

    prediction_embeddings = prediction_embeddings.to("cuda")
    reference_embeddings = reference_embeddings.to("cuda")

    scores = cosine_similarity_pytorch(prediction_embeddings, reference_embeddings)
    return scores

def classify_prediction(prediction: str, good_references: List[str], bad_references: List[str]) -> str:
    """
    Классифицирует предсказание как 'good' или 'bad' на основе сходства с эталонными примерами.
    """
    good_scores = compare_embeddings([prediction], good_references)
    avg_good_score = average_score(good_scores[0])

    bad_scores = compare_embeddings([prediction], bad_references)
    avg_bad_score = average_score(bad_scores[0])

    if avg_good_score > avg_bad_score and avg_good_score > 0.5:
        return "good"
    else:
        return "bad"

collections = ["rag", "good_predictions", "bad_predictions"]
ensure_collections_exist(collections=collections, vector_size=VECTOR_SIZE)
