import chromadb
from src.rag.ingest_chroma import HostedEmbedder, CHROMA_DIR, COLLECTION_NAME

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
col = client.get_collection(COLLECTION_NAME)
embedder = HostedEmbedder()

query = "Mumbai hotel cap for L3 employee in January"
q_vec = embedder.embed_texts([query])[0]

res = col.query(
    query_embeddings=[q_vec],
    n_results=5,
    where={
        "$and": [
            {"status": {"$eq": "ACTIVE"}},
            {"region": {"$in": ["IN", "GLOBAL"]}},
            {"expense_category": {"$eq": "hotel"}},
        ]
    },
)

for cid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
    print(f"{cid} | {meta['policy_ref']} | {meta['grade_band']} | cos_dist={dist:.4f}")