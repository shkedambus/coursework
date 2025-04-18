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
    file_names: List[str]

class UpsertResponse(BaseModel):
    success: bool

class ClearRequest(BaseModel):
    user_id: int

class ClearResponse(BaseModel):
    success: bool
    empty: bool

class DeleteFileRequest(BaseModel):
    user_id: int
    file_name: str

class DeleteFileResponse(BaseModel):
    success: bool
    empty: bool

class CompareRequest(BaseModel):
    question: str
    context: str
    summary: str

class CompareResponse(BaseModel):
    score_question: float
    score_context: float

class ListFilesRequest(BaseModel):
    user_id: int

class ListFilesResponse(BaseModel):
    file_names: List[str]

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

@app.post("/upsert", response_model=UpsertResponse)
async def upsert(req: UpsertRequest):
    try:
        embeddings = embedding_model.encode(
            req.chunks,
            batch_size=8,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        indexer.upsert_chunks(
            file_names=req.file_names,
            chunks=req.chunks,
            embeddings=embeddings,
            user_id=req.user_id
        )
        return UpsertResponse(success=True)
    except Exception as e:
        full_log(logger=logger, where="/upsert")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear", response_model=ClearResponse)
async def clear(req: ClearRequest):
    try:
        empty = indexer.delete_user_data(user_id=req.user_id)
        return ClearResponse(success=True, empty=empty)
    except Exception as e:
        full_log(logger=logger, where="/clear")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_file", response_model=DeleteFileResponse)
async def delete_file(req: DeleteFileRequest):
    try:
        empty = indexer.delete_user_data(user_id=req.user_id, file_name=req.file_name)
        return DeleteFileResponse(success=True, empty=empty)
    except Exception as e:
        full_log(logger=logger, where="/drop")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/drop", response_model=ClearResponse)
async def dtop(req: ClearRequest):
    try:
        indexer.clear_collection()
        return ClearResponse(success=True, empty=False)
    except Exception as e:
        full_log(logger=logger, where="/drop")
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

@app.post("/list_files", response_model=ListFilesResponse)
async def list_files(req: ListFilesRequest):
    try:
        hits = indexer.get_all_chunks(user_id=req.user_id)
        file_names = list(set([h.payload["file_name"] for h in hits]))
        return ListFilesResponse(file_names=file_names)
    except Exception as e:
        full_log(logger=logger, where="/list_files")
        raise HTTPException(status_code=500, detail=str(e))
