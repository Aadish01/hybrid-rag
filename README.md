# Hybrid Search RAG

A document question-answering API that combines:

- Pinecone vector search
- BM25 keyword search
- Reciprocal Rank Fusion (RRF)
- Cohere reranking
- Claude answer generation
- FastAPI

## Requirements

- Python 3.11+
- Pinecone account and API key
- Anthropic API key
- Cohere API key (optional)

## Setup

Create a virtual environment:

    python -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Create `.env` with `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`,
`ANTHROPIC_API_KEY`, and optionally `COHERE_API_KEY`.

## Pinecone setup

Use a document-schema index named `hybrid-rag` with a string field that has
integrated embedding enabled (for example, `body` with `llama-text-embed-v2`).
The app discovers the field and uses the Pinecone Documents API (`2026-07`)
for ingestion and search.

## Add documents

Place PDF, Markdown, and TXT files in `data/`.

## Run the API

    uvicorn app.main:app --reload

Open:

    http://127.0.0.1:8000/docs

## Index documents

Call POST /ingest with:

    {}

## Ask a question

Call POST /ask with:

    {
      "query": "How do I debug a stuck transaction?",
      "top_k": 5
    }

## Health check

Call GET /health.

## Production considerations

- Add authentication and authorization.
- Validate citation support.
- Add document deletion and incremental indexing.
- Use token-aware chunking.
- Add retrieval evaluation and automated tests.
- Restrict ingestion paths and file sizes.
- Add logging, timeouts, and error handling.
