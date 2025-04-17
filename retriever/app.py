import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from qdrant_utils import QdrantIndexer
from sentence_transformers import SentenceTransformer

from shared.logger import get_logger, full_log
logger = get_logger("retriever/app.py")

class RetrieveRequest(BaseModel):
    query: str
    user_id: int

class RetrieveResponse(BaseModel):
    chunks: List[str]

class UpsertRequest(BaseModel):
    user_id: int
    chunks: List[str]

class ClearRequest(BaseModel):
    user_id: int

class SuccessResponse(BaseModel):
    success: bool

app = FastAPI(title="RAG Retriever")

load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
indexer = QdrantIndexer(host=QDRANT_HOST, port=QDRANT_PORT, collection="rag")

embedding_model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True, device="cpu")

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    try:
        query_vector = embedding_model.encode([req.query], normalize_embeddings=True)[0]
        hits = indexer.similarity_search(query_vector=query_vector, user_id=req.user_id)
        return RetrieveResponse(chunks=hits)
    except Exception as e:
        full_log(logger=logger, where="/retrieve")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upsert", response_model=SuccessResponse)
async def upsert(req: UpsertRequest):
    try:
        embeddings = embedding_model.encode(
            req.chunks,
            batch_size=32,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        indexer.upsert_chunks(chunks=req.chunks, embeddings=embeddings, user_id=req.user_id)
        return SuccessResponse(success=True)
    except Exception as e:
        full_log(logger=logger, where="/upsert")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear", response_model=SuccessResponse)
async def clear(req: ClearRequest):
    try:
        indexer.delete_user_data(user_id=req.user_id)
        return SuccessResponse(success=True)
    except Exception as e:
        full_log(logger=logger, where="/clear")
        raise HTTPException(status_code=500, detail=str(e))
