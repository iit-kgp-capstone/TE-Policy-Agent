from types import SimpleNamespace

import chromadb
from fastapi.testclient import TestClient

from app import main
from rag_pipeline.src.rag import hybrid_retriever
from rag_pipeline.src.rag.ingest_chroma import COLLECTION_NAME


def test_human_query_reaches_chroma_and_returns_evidence(tmp_path, monkeypatch):
    index = chromadb.PersistentClient(path=str(tmp_path))
    collection = index.create_collection(COLLECTION_NAME, embedding_function=None)
    collection.add(
        ids=["hotel-1"],
        embeddings=[[1.0, 0.0]],
        documents=["Lodging allowance applies in Mumbai."],
        metadatas=[{
            "status": "ACTIVE",
            "policy_ref": "HOTEL#1",
            "title": "Mumbai (Tier-1, IN) Hotel Cap",
            "region": "IN",
            "grade_band": "ALL",
            "expense_category": "hotel",
            "effective_date": "2026-01-01",
        }],
    )
    collection.add(
        ids=["hotel-2"],
        embeddings=[[0.0, 1.0]],
        documents=["Hotel policy for Pune."],
        metadatas=[{
            "status": "ACTIVE",
            "policy_ref": "HOTEL#2",
            "title": "Pune (Tier-1, IN) Hotel Cap",
            "region": "IN",
            "grade_band": "ALL",
            "expense_category": "hotel",
            "effective_date": "2026-01-01",
        }],
    )
    pune = collection.get(ids=["hotel-2"], include=["metadatas"])["metadatas"][0]
    collection.add(
        ids=["hotel-3"],
        embeddings=[[0.0, 1.0]],
        documents=["Hotel policy for Pune senior employees."],
        metadatas=[{**pune, "policy_ref": "HOTEL#3"}],
    )
    embedded_queries = []

    def embed_texts(texts):
        embedded_queries.extend(texts)
        return [[1.0, 0.0]]

    def rerank(request):
        hits = [{**passage, "score": 0.9} for passage in request.passages]
        if request.query.startswith("Compare"):
            hits.sort(key=lambda hit: hit["id"], reverse=True)
        return hits

    monkeypatch.setattr(
        hybrid_retriever, "HostedEmbedder", lambda **_: SimpleNamespace(embed_texts=embed_texts)
    )
    monkeypatch.setattr(
        hybrid_retriever, "Ranker", lambda **_: SimpleNamespace(rerank=rerank)
    )
    monkeypatch.setattr(
        main, "HybridPolicyRetriever",
        lambda: hybrid_retriever.HybridPolicyRetriever(chroma_dir=tmp_path),
    )

    # The query shares no terms with the stored excerpt, so only dense search can find it.
    query = "How much can I spend overnight?"
    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"status": "ready"}
        response = client.post("/api/v1/retrieve", json={"user_query": query, "region": "IN", "top_k": 1})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "OK"
        assert [item["policy_ref"] for item in body["results"]] == ["HOTEL#1"]
        assert embedded_queries == [query]

        scoped = client.post("/api/v1/retrieve", json={"user_query": "What is the hotel limit in Mumbai?", "grade": "L3"})
        assert scoped.status_code == 200
        assert [item["policy_ref"] for item in scoped.json()["results"]] == ["HOTEL#1"]

        comparison = client.post("/api/v1/retrieve", json={
            "user_query": "Compare hotel limits in Mumbai and Pune", "top_k": 2,
        })
        assert comparison.status_code == 200
        assert {item["policy_ref"] for item in comparison.json()["results"]} == {"HOTEL#1", "HOTEL#3"}

        empty = client.post("/api/v1/retrieve", json={"user_query": query, "region": "US"})
        assert empty.json() == {"status": "NO_POLICY_EVIDENCE", "results": []}
        invalid = client.post("/api/v1/retrieve", json={"user_query": "  "})
        assert invalid.status_code == 422
