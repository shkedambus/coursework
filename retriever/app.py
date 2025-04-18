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

class CompareRequest(BaseModel):
    question: str
    context: str
    summary: str

class CompareResponse(BaseModel):
    score_question: float
    score_context: float

app = FastAPI(title="RAG Retriever")

load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
indexer = QdrantIndexer(host=QDRANT_HOST, port=QDRANT_PORT, collection="rag")

embedding_model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True, device="cpu")

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    try:
        query_vector = embedding_model.encode([req.query], normalize_embeddings=True, prompt_name="query")[0]
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
            batch_size=8,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=True
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

@app.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    try:
        embeddings = embedding_model.encode(
            [req.question, req.context, req.summary],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        question_embeddings, context_embeddings, summary_embeddings = embeddings

        score_question = round(float(indexer.cosine_similarity_pytorch(tensor1=question_embeddings, tensor2=summary_embeddings)), 4)
        score_context = round(float(indexer.cosine_similarity_pytorch(tensor1=context_embeddings, tensor2=summary_embeddings)), 4)

        return CompareResponse(score_question=score_question, score_context=score_context)
    except Exception as e:
        full_log(logger=logger, where="/compare")
        raise HTTPException(status_code=500, detail=str(e))
