"""Pinecone Documents API for indexes with integrated embeddings."""

from functools import lru_cache
from urllib.parse import quote

import httpx

from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME


NAMESPACE = "hybrid-rag"
API_VERSION = "2026-07"
BATCH_SIZE = 96


def _request(method: str, url: str, **kwargs) -> dict:
    if not PINECONE_API_KEY:
        raise RuntimeError("Set PINECONE_API_KEY in your .env file.")

    try:
        response = httpx.request(
            method,
            url,
            headers={
                "Api-Key": PINECONE_API_KEY,
                "X-Pinecone-Api-Version": API_VERSION,
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
            **kwargs,
        )
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Cannot reach Pinecone at {httpx.URL(url).host}. "
            "Check your network and DNS connection."
        ) from exc

    if response.is_error:
        try:
            detail = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            detail = response.text
        raise RuntimeError(f"Pinecone ({response.status_code}): {detail}")

    return response.json()


@lru_cache(maxsize=1)
def get_index() -> dict:
    """Discover the host and embedding field without guessing its name."""
    if not PINECONE_INDEX_NAME:
        raise RuntimeError("Set PINECONE_INDEX_NAME in your .env file.")

    description = _request(
        "GET",
        f"https://api.pinecone.io/indexes/{quote(PINECONE_INDEX_NAME, safe='')}",
    )
    fields = (description.get("schema") or {}).get("fields", {})
    embed_fields = [
        name for name, field in fields.items()
        if field.get("type") == "string" and field.get("embed") is not None
        and field.get("embed") is not False
    ]
    if len(embed_fields) != 1:
        raise RuntimeError(
            f"Index '{PINECONE_INDEX_NAME}' must have one string field with "
            "dense integrated embedding (schema.fields.<field>.embed). "
            "Check the index schema in Pinecone."
        )

    host = description["host"].removeprefix("https://").rstrip("/")
    return {"host": host, "text_field": embed_fields[0]}


def _documents_url(index: dict, operation: str) -> str:
    return (
        f"https://{index['host']}/namespaces/{quote(NAMESPACE, safe='')}"
        f"/documents/{operation}"
    )


def upsert_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return

    index = get_index()
    for start in range(0, len(chunks), BATCH_SIZE):
        documents = []
        for chunk in chunks[start:start + BATCH_SIZE]:
            document = {
                "_id": chunk["chunk_id"],
                "source": chunk["source"],
                "chunk_index": int(chunk.get("chunk_index", 0)),
            }
            if chunk.get("page") is not None:
                document["page"] = int(chunk["page"])
            document[index["text_field"]] = chunk["text"]
            documents.append(document)

        response = _request(
            "POST", _documents_url(index, "upsert"),
            json={"documents": documents},
        )
        if response.get("upserted_count") != len(documents):
            raise RuntimeError(
                "Pinecone did not acknowledge every chunk in the batch; "
                "the local corpus was not updated. Retry ingestion."
            )


def semantic_search(query: str, top_k: int = 20) -> list[str]:
    index = get_index()
    response = _request(
        "POST", _documents_url(index, "search"),
        json={
            "top_k": top_k,
            "score_by": [{
                "type": "embed",
                "field": index["text_field"],
                "query": query,
            }],
        },
    )
    return [match["_id"] for match in response["matches"]]
