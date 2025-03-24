from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import hashlib
import uuid
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Qdrant")

# Эмбеддинг-модель
model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
embedding_model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")

# Подключаемся к локальному Qdrant
qdrant_client = QdrantClient("localhost", port=6333)

collections = ["main", "good_predictions", "bad_predictions"]
vector_size = 1536

for collection in collections:
    if not qdrant_client.collection_exists(collection):
        qdrant_client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def delete_collection(collection_name):
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
        print(f"Коллекция '{collection_name}' удалена.")
    else:
        print(f"Коллекция '{collection_name}' не существует.")

def clear_collection(collection_name):
    if not qdrant_client.collection_exists(collection_name):
        print(f"Коллекция '{collection_name}' не существует.")
        return

    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=Filter(must=[])
    )

    print(f"Все данные в коллекции '{collection_name}' удалены.")

def delete_user_data(collection_name, user_id):
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector={"filter": {"must": [{"key": "user_id", "match": {"value": user_id}}]}}
        )
        print(f"Все данные пользователя {user_id} удалены из коллекции '{collection_name}'.")
    else:
        print(f"Коллекция '{collection_name}' не существует.")

def inspect_collection(collection_name, limit=10):
    if not qdrant_client.collection_exists(collection_name):
        print(f"Коллекция '{collection_name}' не существует.")
        return

    points = qdrant_client.scroll(
        collection_name=collection_name,
        limit=limit
    )[0]

    if not points:
        print(f"Коллекция '{collection_name}' пуста.")
        return
    
    print(f"Данные в коллекции '{collection_name}':")
    for point in points:
        print(f"ID: {point.id}, User ID: {point.payload.get('user_id', 'N/A')}")
        print(f"Текст: {point.payload.get('text', 'Нет текста')}")
        print("-" * 50)

def embedd_chunks(chunks):
    chunks = list(set(chunks))

    embeddings = embedding_model.encode(
        chunks,
        batch_size=32,
        convert_to_tensor=True,    # Ускоряет на GPU
        normalize_embeddings=True  # Ускоряет поиск по cosine similarity
    )

    return embeddings

# Функция для создания уникального хэша для текста чанка
def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Функция для проверки, есть ли уже такой хэш в базе
def is_duplicate_chunk(collection_name, chunk_hash):
    existing_chunks, _ = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter={"must": [{"key": "chunk_hash", "match": {"value": chunk_hash}}]},
        limit=1
    )
    return len(existing_chunks) > 0

def update_db(collection_name, chunks, embeddings, user_id):
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
        print(f"Добавлено {len(unique_chunks)} новых чанков для пользователя {user_id}")
    else:
        print("Все чанки уже есть в базе, ничего не добавлено.")

def get_chunks(collection_name, query, user_id=0):
    query_vector = embedding_model.encode([query])[0]
    retrieved_chunks = similarity_search(collection_name, query_vector, user_id)
    return retrieved_chunks

def similarity_search(collection_name, query_vector, user_id, threshold=0.3, top_k=10):
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=threshold,
        query_filter={"must": [{"key": "user_id", "match": {"value": user_id}}]}
    )

    # for hit in search_result:
    #     print(f"Найден чанк (score={hit.score}, len={len(hit.payload['text'])}): {hit.payload['text']}")

    retrieved_texts = [hit.payload["text"] for hit in search_result]
    return retrieved_texts

def average_score(scores):
    if isinstance(scores[0], list):  # Если scores — матрица
        flat_scores = [score for row in scores for score in row]  # Разворачиваем в одномерный список
    else:
        flat_scores = scores

    return round(float(sum(flat_scores) / len(flat_scores)), 3)

def cosine_similarity_pytorch(tensor1, tensor2):
    # Нормализуем векторы
    tensor1_normalized = tensor1 / tensor1.norm(dim=1, keepdim=True)
    tensor2_normalized = tensor2 / tensor2.norm(dim=1, keepdim=True)

    # Вычисляем косинусное сходство
    cosine_sim = torch.mm(tensor1_normalized, tensor2_normalized.transpose(0, 1))
    return cosine_sim

def compare_embeddings(predictions, references):
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

def classify_prediction(prediction, good_references, bad_references):
    good_scores = compare_embeddings([prediction], good_references)
    avg_good_score = average_score(good_scores[0])

    bad_scores = compare_embeddings([prediction], bad_references)
    avg_bad_score = average_score(bad_scores[0])

    if avg_good_score > avg_bad_score and avg_good_score > 0.5:
        return "good"
    else:
        return "bad"