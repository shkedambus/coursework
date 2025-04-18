import torch
import hashlib, uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, VectorParams, PointStruct, PointIdsList, MatchValue, FieldCondition

from shared.logger import get_logger
logger = get_logger("retriever/qdrant_utils")

class QdrantIndexer:
    def __init__(self, host, port, collection):
        self.client = QdrantClient(host=host, port=port)
        if not self.client.collection_exists(collection):
            self.client.create_collection(
              collection_name=collection,
              vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        self.collection = collection

    def hash_text(self, text: str) -> str:
        """
        Создает уникальный хэш для текста чанка с использованием алгоритма SHA-256.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def get_existing_chunk(self, chunk_hash: str):
        """
        Возвращает существующий чанк по заданному хэшу или None, если такого чанка нет.
        """
        existing_chunks, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter={"must": [{"key": "chunk_hash", "match": {"value": chunk_hash}}]},
            limit=1
        )
        if existing_chunks:
            return existing_chunks[0]
        return None

    def upsert_chunks(self, chunks: List[str], embeddings: torch.Tensor, user_id: int) -> None:
        """
        Добавляет уникальные чанки и их эмбеддинги в коллекцию Qdrant, 
        используя метаданные в виде списка user_ids.
        Если чанк уже существует, и пользователь не привязан к нему, добавляет его в список.
        """
        new_points = []
        updated_points = []

        for chunk, emb in zip(chunks, embeddings):
            chunk_hash = self.hash_text(chunk)
            existing_chunk = self.get_existing_chunk(chunk_hash)

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
            self.client.upsert(collection_name=self.collection, points=new_points)
            logger.info(f"Добавлено {len(new_points)} новых чанков для пользователя {user_id}")

        if updated_points:
            self.client.upsert(collection_name=self.collection, points=updated_points)
            logger.info(f"Обновлено {len(updated_points)} существующих чанков для пользователя {user_id}")

    def similarity_search(self, query_vector: torch.Tensor, user_id: int, threshold: float = 0.5, top_k: int = 5) -> List[str]:
        """
        Выполняет поиск ближайших чанков в коллекции по косинусному сходству и возвращает тексты.
        Результаты фильтруются по тому, чтобы заданный user_id присутствовал в списке user_ids.
        """
        search_result = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=threshold,
            # query_filter={"must": [{"key": "user_ids", "match": {"value": user_id}}]}
            query_filter=Filter(must=[{"key": "user_ids", "match": {"value": user_id}}])
        )

        retrieved_texts = [hit.payload["text"] for hit in search_result]

        # for hit in search_result:
        #     logger.info(f"Найден чанк (score={hit.score}, len={len(hit.payload['text'])}):\n{hit.payload['text']}")

        return retrieved_texts
    
    def delete_collection(self) -> None:
        """
        Удаляет указанную коллекцию из базы Qdrant, если она существует.
        """
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
            logger.info(f"Коллекция '{self.collection}' удалена.")
        else:
            logger.info(f"Коллекция '{self.collection}' не существует.")

    def clear_collection(self) -> None:
        """
        Удаляет все объекты в коллекции, оставляя саму коллекцию пустой.
        """
        if not self.client.collection_exists(self.collection):
            logger.info(f"Коллекция '{self.collection}' не существует.")
            return

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[])
        )

        logger.info(f"Все данные в коллекции '{self.collection}' удалены.")

    def get_all_chunks(self, user_id: int) -> list:
        """
        Возвращает все чанки, связанные с указанным user_id.
        """
        all_hits = []
        batch_limit = 100
        next_page_token = None

        q_filter = None
        if user_id != -1:
            q_filter = Filter(must=[
                FieldCondition(
                    key="user_ids",
                    match=MatchValue(value=user_id)
                )
            ])

        while True:
            hits, next_page_token = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=q_filter,
                limit=batch_limit,
                offset=next_page_token,
                with_vectors=True,
                with_payload=True
            )
                
            all_hits.extend(hits)
            
            if not next_page_token or len(hits) < batch_limit:
                break

        return all_hits

    def delete_user_data(self, user_id: int) -> None:
        """
        Удаляет данные, связанные с конкретным пользователем, из указанной коллекции.
        Для каждого найденного чанка:
        - если чанк принадлежит нескольким пользователям, удаляет user_id из списка user_ids
        - если чанк принадлежит только данному пользователю, удаляет весь чанк из коллекции
        """
        if not self.client.collection_exists(self.collection):
            logger.info(f"Коллекция '{self.collection}' не существует.")
            return

        hits = self.get_all_chunks(user_id=user_id)
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
                    self.client.delete(
                        collection_name=self.collection,
                        points_selector=PointIdsList(points=[hit.id])
                    )
                    # logger.info(f"Удалён чанк {hit.id}, принадлежащий пользователю {user_id}")

        if updated_points:
            self.client.upsert(
                collection_name=self.collection,
                points=updated_points
            )

        logger.info(f"Обработка удаления данных пользователя {user_id} завершена в коллекции '{self.collection}'.")

    def inspect_collection(self, user_id: int = -1) -> None:
        """
        Выводит информацию о содержимом коллекции - ID пользователей и связанные с ними тексты.
        """
        if not self.client.collection_exists(self.collection):
            logger.info(f"Коллекция '{self.collection}' не существует.")
            return

        points = self.get_all_chunks(user_id=user_id)

        if not points:
            logger.info(f"Коллекция '{self.collection}' пуста.")
            return
        
        logger.info(f"Данные в коллекции '{self.collection}':")
        for point in points:
            logger.info(f"ID: {point.id}, User IDs: {point.payload.get('user_ids', 'N/A')}")
            logger.info(f"Текст: {point.payload.get('text', 'Нет текста')}")
            logger.info("-" * 50)

    def cosine_similarity_pytorch(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет косинусное сходство между двумя тензорами, используя PyTorch.
        """     
        if len(tensor1.shape) == 1:
            tensor1 = tensor1.unsqueeze(0)
        if len(tensor2.shape) == 1:
            tensor2 = tensor2.unsqueeze(0)

        return torch.cosine_similarity(tensor1, tensor2)
