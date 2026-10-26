import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME", "hybrid-rag"
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv(
    "CLAUDE_MODEL", "claude-sonnet-4-6"
)

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = os.getenv(
    "COHERE_RERANK_MODEL", "rerank-v4.0-pro"
)

DATA_DIR = os.getenv("DATA_DIR", "data")

CHUNKS_FILE = os.path.join(
    DATA_DIR, "indexed_chunks.json"
)
