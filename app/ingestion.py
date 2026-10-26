import hashlib
import json
from pathlib import Path
from typing import Optional
from pypdf import PdfReader

from app.config import DATA_DIR, CHUNKS_FILE
from app.vector_store import upsert_chunks


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def read_document(path: Path) -> list[dict]:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        sections = []

        for page_num, page in enumerate(
            reader.pages, start=1
        ):
            text = (page.extract_text() or "").strip()

            if text:
                sections.append({
                    "text": text,
                    "page": page_num,
                })

        return sections

    if path.suffix.lower() in {".md", ".txt"}:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).strip()

        return (
            [{"text": text, "page": None}]
            if text else []
        )

    return []


def chunk_text(
    text: str,
    chunk_size: int = 3500,
    overlap: int = 400,
) -> list[str]:
    """Split text by characters with overlapping context."""
    if chunk_size <= overlap:
        raise ValueError(
            "chunk_size must be greater than overlap"
        )

    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def load_existing_chunks() -> list[dict]:
    path = Path(CHUNKS_FILE)

    if not path.exists():
        return []

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_chunks(chunks: list[dict]) -> None:
    path = Path(CHUNKS_FILE)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    path.write_text(
        json.dumps(
            chunks,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def ingest_directory(
    directory: Optional[str] = None,
) -> dict:
    root = Path(directory or DATA_DIR)

    if not root.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {root}"
        )

    chunks = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        for section in read_document(path):
            text_chunks = chunk_text(section["text"])

            for index, text in enumerate(text_chunks):
                raw_id = (
                    f"{path.as_posix()}:"
                    f"{section.get('page')}:"
                    f"{index}:{text}"
                )

                chunk_id = hashlib.sha256(
                    raw_id.encode("utf-8")
                ).hexdigest()[:32]

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": str(path),
                    "page": section.get("page"),
                    "chunk_index": index,
                })

    if not chunks:
        return {
            "documents_indexed": 0,
            "chunks_indexed": 0,
            "message": "No supported documents found.",
        }

    upsert_chunks(chunks)
    save_chunks(chunks)

    return {
        "documents_indexed": len({
            chunk["source"] for chunk in chunks
        }),
        "chunks_indexed": len(chunks),
        "message": "Ingestion complete.",
    }
