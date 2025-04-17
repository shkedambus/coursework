import os
import hashlib
import uuid
from typing import List
from dotenv import load_dotenv

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, VectorParams, PointStruct, PointIdsList
from sentence_transformers import SentenceTransformer

# Эмбеддинг-модель
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
VECTOR_SIZE = 1536
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True, device="cpu")

# device = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()

# Подключаемся к Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
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

def get_all_chunks(collection_name: str, user_id: int) -> List:
    """
    Получает все чанки из коллекции, в которых user_id содержится в списке user_ids.
    """
    all_hits = []
    next_page_token = None

    batch_limit = 100

    scroll_filter = None
    if user_id != -1:
        scroll_filter = {"must": [{"key": "user_ids", "match": {"value": user_id}}]}

    while True:
        # Если next_page_token не None, добавляем его в параметры запроса
        scroll_params = {
            "collection_name": collection_name,
            "scroll_filter": scroll_filter,
            "limit": batch_limit,
            "with_vectors": True
        }
        if next_page_token:
            scroll_params["next_page_token"] = next_page_token

        hits, next_page_token = qdrant_client.scroll(**scroll_params)
        all_hits.extend(hits)

        # Если возвращено меньше записей, чем лимит, или нет следующей страницы, значит данные закончились
        if not next_page_token or len(hits) < batch_limit:
            break

    return all_hits

def delete_user_data(collection_name: str, user_id: int) -> None:
    """
    Удаляет данные, связанные с конкретным пользователем, из указанной коллекции.
    Для каждого найденного чанка:
      - если чанк принадлежит нескольким пользователям, удаляет user_id из списка user_ids
      - если чанк принадлежит только данному пользователю, удаляет весь чанк из коллекции
    """
    if not qdrant_client.collection_exists(collection_name):
        logger.info(f"Коллекция '{collection_name}' не существует.")
        return

    hits = get_all_chunks(collection_name=collection_name, user_id=user_id)
    updated_points = []

    for hit in hits:
        payload = hit.payload
        user_ids = payload.get("user_ids", [])
        if user_id in user_ids:
            if len(user_ids) > 1:
                # Если чанк принадлежит нескольким пользователям, удаляем заданного пользователя из списка
                user_ids.remove(user_id)
                payload["user_ids"] = user_ids
                updated_point = PointStruct(id=hit.id, vector=hit.vector, payload=payload)
                updated_points.append(updated_point)
            else:
                # Если чанк принадлежит только данному пользователю, удаляем его
                qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=PointIdsList(points=[hit.id])
                )
                logger.info(f"Удалён чанк {hit.id}, принадлежащий пользователю {user_id}")

    # Обновляем чанки в базе
    if updated_points:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=updated_points
        )

    logger.info(f"Обработка удаления данных пользователя {user_id} завершена в коллекции '{collection_name}'.")

def inspect_collection(collection_name: str, user_id: int = -1) -> None:
    """
    Выводит информацию о содержимом коллекции - ID пользователей и связанные с ними тексты.
    """
    if not qdrant_client.collection_exists(collection_name):
        logger.info(f"Коллекция '{collection_name}' не существует.")
        return

    points = get_all_chunks(collection_name=collection_name, user_id=user_id)

    if not points:
        logger.info(f"Коллекция '{collection_name}' пуста.")
        return
    
    logger.info(f"Данные в коллекции '{collection_name}':")
    for point in points:
        logger.info(f"ID: {point.id}, User IDs: {point.payload.get('user_ids', 'N/A')}")
        logger.info(f"Текст: {point.payload.get('text', 'Нет текста')}")
        logger.info("-" * 50)

def embedd_chunks(chunks: List[str]) -> torch.Tensor:
    """
    Строит эмбеддинги для уникальных текстовых чанков с помощью модели Qwen и возвращает тензор.
    """
    chunks = list(set(chunks))

    # embedding_model.to(device)

    embeddings = embedding_model.encode(
        chunks,
        batch_size=32,
        convert_to_tensor=True,    # Ускоряет на GPU
        normalize_embeddings=True  # Ускоряет поиск по cosine similarity
    )

    # embedding_model.to("cpu")
    # torch.cuda.empty_cache()

    return embeddings

def hash_text(text: str) -> str:
    """
    Создает уникальный хэш для текста чанка с использованием алгоритма SHA-256.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_existing_chunk(collection_name: str, chunk_hash: str):
    """
    Возвращает существующий чанк по заданному хэшу или None, если такого чанка нет.
    """
    existing_chunks, _ = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter={"must": [{"key": "chunk_hash", "match": {"value": chunk_hash}}]},
        limit=1
    )
    if existing_chunks:
        return existing_chunks[0]
    return None

def update_db(collection_name: str, chunks: List[str], embeddings: torch.Tensor, user_id: int) -> None:
    """
    Добавляет уникальные чанки и их эмбеддинги в коллекцию Qdrant, 
    используя метаданные в виде списка user_ids.
    Если чанк уже существует, и пользователь не привязан к нему, добавляет его в список.
    """
    new_points = []
    updated_points = []

    for chunk, emb in zip(chunks, embeddings):
        chunk_hash = hash_text(chunk)
        existing_chunk = get_existing_chunk(collection_name, chunk_hash)

        if existing_chunk:
            # Проверяем, имеется ли user_ids, и добавляем, если current user_id отсутствует
            user_ids = existing_chunk.payload.get("user_ids", [])
            if user_id not in user_ids:
                user_ids.append(user_id)
                existing_chunk.payload["user_ids"] = user_ids
                updated_point = PointStruct(id=existing_chunk.id, vector=emb, payload=existing_chunk.payload)
                updated_points.append(updated_point)
        else:
            # Чанк отсутствует - создаем новый с user_ids в виде списка
            point_id = int(uuid.uuid4().int % 1e9)
            payload = {"text": chunk, "user_ids": [user_id], "chunk_hash": chunk_hash}
            new_point = PointStruct(id=point_id, vector=emb, payload=payload)
            new_points.append(new_point)

    if new_points:
        qdrant_client.upsert(collection_name=collection_name, points=new_points)
        logger.info(f"Добавлено {len(new_points)} новых чанков для пользователя {user_id}")

    if updated_points:
        qdrant_client.upsert(collection_name=collection_name, points=updated_points)
        logger.info(f"Обновлено {len(updated_points)} существующих чанков для пользователя {user_id}")

def get_relevant_chunks(collection_name: str, query: str, user_id: int = 0) -> List[str]:
    """
    Находит наиболее релевантные чанки из коллекции по эмбеддингу запроса.
    """
    # embedding_model.to(device)

    query_vector = embedding_model.encode(
        [query],
        convert_to_tensor=True,    # Ускоряет на GPU
        normalize_embeddings=True  # Ускоряет поиск по cosine similarity
    )[0]

    # embedding_model.to("cpu")
    # torch.cuda.empty_cache()

    retrieved_chunks = similarity_search(collection_name, query_vector, user_id)
    return retrieved_chunks

def similarity_search(collection_name: str, query_vector: torch.Tensor, user_id: int, threshold: float = 0.3, top_k: int = 10) -> List[str]:
    """
    Выполняет поиск ближайших чанков в коллекции по косинусному сходству и возвращает тексты.
    Результаты фильтруются по тому, чтобы заданный user_id присутствовал в списке user_ids.
    """
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=threshold,
        query_filter={"must": [{"key": "user_ids", "match": {"value": user_id}}]}
    )

    retrieved_texts = [hit.payload["text"] for hit in search_result]

    for hit in search_result:
        logger.info(f"Найден чанк (score={hit.score}, len={len(hit.payload['text'])}):\n{hit.payload['text']}")

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
    # tensor1_normalized = tensor1 / tensor1.norm(dim=1, keepdim=True)
    # tensor2_normalized = tensor2 / tensor2.norm(dim=1, keepdim=True)

    # Вычисляем косинусное сходство
    cosine_sim = torch.mm(tensor1, tensor2.transpose(0, 1))
    return cosine_sim

def compare_embeddings(predictions: List[str], references: List[str]) -> torch.Tensor:
    """
    Сравнивает эмбеддинги предсказаний и эталонных ответов, возвращая матрицу сходства.
    """
    # embedding_model.to(device)

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

    # prediction_embeddings = prediction_embeddings.to(device)
    # reference_embeddings = reference_embeddings.to(device)

    # embedding_model.to("cpu")
    # torch.cuda.empty_cache()

    scores = cosine_similarity_pytorch(prediction_embeddings, reference_embeddings)

    # prediction_embeddings = prediction_embeddings.to("cpu")
    # reference_embeddings = reference_embeddings.to("cpu")

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
