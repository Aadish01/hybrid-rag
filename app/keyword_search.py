import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-zA-Z0-9_./:-]+",
        text.lower(),
    )


class KeywordSearch:
    def __init__(self, chunks: list[dict]):
        self.ids = [
            chunk["chunk_id"]
            for chunk in chunks
        ]

        tokenized_documents = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = (
            BM25Okapi(tokenized_documents)
            if tokenized_documents
            else None
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[str]:
        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(
            tokenize(query)
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        return [
            self.ids[i]
            for i in ranked_indices[:top_k]
            if scores[i] > 0
        ]
