from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.ingestion import (
    load_existing_chunks,
    ingest_directory,
)
from app.keyword_search import KeywordSearch
from app.retrieval import retrieve
from app.generation import generate_answer


app = FastAPI(
    title="Hybrid Search RAG",
    description=(
        "Hybrid semantic and keyword search "
        "with reranking and cited answers."
    ),
    version="1.0.0",
)


# Load the local corpus and build BM25 at startup.
_chunks = load_existing_chunks()
_keyword_search = KeywordSearch(_chunks)


class AskRequest(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=4000,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class IngestRequest(BaseModel):
    directory: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": len(_chunks),
    }


@app.post("/ingest")
def ingest(request: IngestRequest):
    global _chunks, _keyword_search

    try:
        result = ingest_directory(
            request.directory
        )

        _chunks = load_existing_chunks()
        _keyword_search = KeywordSearch(_chunks)

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/ask")
def ask(request: AskRequest):
    if not _chunks:
        raise HTTPException(
            status_code=400,
            detail=(
                "No indexed chunks. Add documents "
                "to data/ and call POST /ingest first."
            ),
        )

    try:
        sources = retrieve(
            request.query,
            _chunks,
            _keyword_search,
            top_k=request.top_k,
        )

        answer = generate_answer(
            request.query,
            sources,
        )

        return {
            "query": request.query,
            "answer": answer,
            "sources": [
                {
                    "citation": f"[{index}]",
                    "source": chunk["source"],
                    "page": chunk.get("page"),
                    "chunk_id": chunk["chunk_id"],
                    "relevance_score": chunk.get(
                        "relevance_score"
                    ),
                }
                for index, chunk in enumerate(
                    sources,
                    start=1,
                )
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
