from app.vector_store import semantic_search
from app.keyword_search import KeywordSearch

from app.config import (
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
)


def reciprocal_rank_fusion(
    *ranked_lists: list[str],
    k0: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}

    for results in ranked_lists:
        seen = set()

        for rank, doc_id in enumerate(
            results, start=1
        ):
            if doc_id in seen:
                continue

            seen.add(doc_id)

            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + 1.0 / (k0 + rank)
            )

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


def rerank(
    query: str,
    chunks: list[dict],
    top_n: int = 5,
) -> list[dict]:
    if not chunks:
        return []

    if not COHERE_API_KEY:
        return chunks[:top_n]

    import cohere

    client = cohere.ClientV2(
        api_key=COHERE_API_KEY
    )

    response = client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=[
            chunk["text"] for chunk in chunks
        ],
        top_n=min(top_n, len(chunks)),
    )

    results = []

    for result in response.results:
        chunk = dict(chunks[result.index])
        chunk["relevance_score"] = float(
            result.relevance_score
        )
        results.append(chunk)

    return results


def retrieve(
    query: str,
    chunks: list[dict],
    keyword_search: KeywordSearch,
    top_k: int = 5,
) -> list[dict]:
    if not chunks:
        return []

    vector_ids = semantic_search(query, top_k=20)

    keyword_ids = keyword_search.search(
        query,
        top_k=20,
    )

    fused = reciprocal_rank_fusion(
        vector_ids,
        keyword_ids,
    )

    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    candidates = [
        chunks_by_id[doc_id]
        for doc_id, _ in fused
        if doc_id in chunks_by_id
    ][:20]

    return rerank(
        query,
        candidates,
        top_n=top_k,
    )
