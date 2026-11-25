import hashlib
import json
import os
import ssl
from pathlib import Path
from typing import Any

import chromadb
import truststore
from dotenv import load_dotenv
from openai import DefaultHttpxClient, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .chunker import chunk_policy_document
from .metadata import build_chroma_metadata

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = REPO_ROOT / "data/corpus/te_policy_corpus.jsonl"
CHROMA_DIR = REPO_ROOT / "data/chroma_db"
EMBED_CACHE_PATH = REPO_ROOT / "data/cache/embedding_cache.json"
COLLECTION_NAME = "te_policy_chunks"
BATCH_SIZE = 16


class HostedEmbedder:
    def __init__(self, cache_path: Path | None = EMBED_CACHE_PATH):
        load_dotenv(REPO_ROOT / ".env")
        api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set EMBEDDING_API_KEY or OPENAI_API_KEY.")
        truststore.inject_into_ssl()
        self.model = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        ca_file = os.getenv("SSL_CERT_FILE")
        http_client = (
            DefaultHttpxClient(verify=ssl.create_default_context(cafile=ca_file), timeout=10.0)
            if ca_file else None
        )
        self.client = OpenAI(
            base_url=os.getenv("EMBEDDING_BASE_URL") or "https://api.openai.com/v1",
            api_key=api_key,
            timeout=10.0,
            max_retries=0,
            http_client=http_client,
        )
        self.cache_path = cache_path
        self.cache: dict[str, list[float]] = (
            json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_path is not None and cache_path.exists()
            else {}
        )

    def _hash(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}::{text}".encode("utf-8")).hexdigest()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=16))
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.cache_path is None:
            return self._embed(texts)
        keys = [self._hash(t) for t in texts]
        missing_indices = [i for i, k in enumerate(keys) if k not in self.cache]

        for start in range(0, len(missing_indices), BATCH_SIZE):
            batch_idx = missing_indices[start : start + BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_idx]
            vectors = self._call_api(batch_texts)
            for idx, vec in zip(batch_idx, vectors):
                self.cache[keys[idx]] = vec

        if missing_indices:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

        return [self.cache[k] for k in keys]


def ingest_corpus() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
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
