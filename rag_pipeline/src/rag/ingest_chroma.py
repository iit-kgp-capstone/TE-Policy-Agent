import hashlib
import json
import os
from pathlib import Path
from typing import Any

import chromadb
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.rag.chunker import chunk_policy_document
from src.rag.metadata import build_chroma_metadata


from dotenv import dotenv_values, find_dotenv

from flask import Flask
import truststore
truststore.inject_into_ssl()


app = Flask(__name__)

# # Locate the .env file in the parent directory
# env_path = Path(__file__).resolve().parent.parent / ".env"

# # Load the variables as a dictionary and update app.config
# config_values = dotenv_values(env_path)
# app.config.update(config_values)

# # Hosted Embeddings Configuration (OpenAI-compatible: OpenAI / Together / OpenRouter / vLLM)
# EMBEDDING_BASE_URL = config_values.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
# EMBEDDING_API_KEY = config_values.get("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
# EMBEDDING_MODEL = config_values.get("EMBEDDING_MODEL", "text-embedding-3-small")

# Automatically finds the nearest .env walking upward
env_path = find_dotenv(usecwd=True) or str(Path(__file__).resolve().parents[2] / ".env")
print("Loading .env from:", env_path, "| Exists:", Path(env_path).exists())

config_values = dotenv_values(env_path)
print("Keys found in .env:", list(config_values.keys()))  # Safe: prints names only, not values

EMBEDDING_BASE_URL = config_values.get("EMBEDDING_BASE_URL") or "https://api.openai.com/v1"
EMBEDDING_API_KEY = (
    config_values.get("EMBEDDING_API_KEY")
    or config_values.get("OPENAI_API_KEY")
    or os.getenv("EMBEDDING_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
EMBEDDING_MODEL = config_values.get("EMBEDDING_MODEL") or "text-embedding-3-small"
SSL_CERT_FILE = config_values.get("SSL_CERT_FILE") or os.getenv("SSL_CERT_FILE")
CORPUS_PATH = Path("../data/corpus/te_policy_corpus.jsonl")
CHROMA_DIR = Path("../data/chroma_db")
EMBED_CACHE_PATH = Path("../data/cache/embedding_cache.json")
COLLECTION_NAME = "te_policy_chunks"

if not EMBEDDING_API_KEY:
    raise ValueError(f"No API key found in {env_path}. Check your .env variable name.")

BATCH_SIZE = 16

print(EMBEDDING_BASE_URL+ "  " + EMBEDDING_MODEL)

class HostedEmbedder:
    def __init__(self, cache_path: Path = EMBED_CACHE_PATH):
        self.client = OpenAI(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY)
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, list[float]] = (
            json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_path.exists()
            else {}
        )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(f"{EMBEDDING_MODEL}::{text}".encode("utf-8")).hexdigest()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=16))
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in resp.data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        keys = [self._hash(t) for t in texts]
        missing_indices = [i for i, k in enumerate(keys) if k not in self.cache]

        for start in range(0, len(missing_indices), BATCH_SIZE):
            batch_idx = missing_indices[start : start + BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_idx]
            vectors = self._call_api(batch_texts)
            for idx, vec in zip(batch_idx, vectors):
                self.cache[keys[idx]] = vec

        if missing_indices:
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

        return [self.cache[k] for k in keys]


def ingest_corpus() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            chunks = chunk_policy_document(doc, min_tokens=80, max_tokens=450)
            for idx, ch in enumerate(chunks):
                ids.append(ch.chunk_id)
                documents.append(ch.text)
                metadatas.append(
                    build_chroma_metadata(doc, idx, ch.has_table, ch.token_estimate)
                )

    embedder = HostedEmbedder()
    embeddings = embedder.embed_texts(documents)

    # Upsert in batches to stay well within ChromaDB batch limits
    for start in range(0, len(ids), 200):
        end = start + 200
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    print(f"Ingested {len(ids)} chunks into ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    ingest_corpus()