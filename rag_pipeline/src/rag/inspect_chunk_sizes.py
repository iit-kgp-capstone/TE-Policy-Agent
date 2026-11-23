import statistics
from pathlib import Path
import chromadb

# Adjust parents[2] or parents[3] to match where data/chroma_db lives
#PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = str(Path(__file__).resolve().parents[3]  / "data/chroma_db")
COLLECTION_NAME = "te_policy_chunks"

def inspect_chunk_sizes() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    data = collection.get(include=["documents", "metadatas"])
    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]

    if not docs:
        print("Collection is empty.")
        return

    token_sizes = [m.get("token_estimate", int(len(d.split()) * 1.33)) for d, m in zip(docs, metas)]
    word_sizes = [len(d.split()) for d in docs]
    char_sizes = [len(d) for d in docs]

    min_idx = token_sizes.index(min(token_sizes))
    max_idx = token_sizes.index(max(token_sizes))

    print(f"Total chunks: {len(docs)}\n")
    print(f"{'Metric':<14} {'Min':>8} {'Max':>8} {'Mean':>8} {'Median':>8}")
    print("-" * 50)
    print(f"{'Tokens (est)':<14} {min(token_sizes):>8} {max(token_sizes):>8} {statistics.mean(token_sizes):>8.1f} {statistics.median(token_sizes):>8.1f}")
    print(f"{'Words':<14} {min(word_sizes):>8} {max(word_sizes):>8} {statistics.mean(word_sizes):>8.1f} {statistics.median(word_sizes):>8.1f}")
    print(f"{'Characters':<14} {min(char_sizes):>8} {max(char_sizes):>8} {statistics.mean(char_sizes):>8.1f} {statistics.median(char_sizes):>8.1f}")

    print("\nShortest chunk:")
    print(f"  ID: {ids[min_idx]} | Tokens: {token_sizes[min_idx]} | Title: {metas[min_idx].get('title')}")
    print("Longest chunk:")
    print(f"  ID: {ids[max_idx]} | Tokens: {token_sizes[max_idx]} | Title: {metas[max_idx].get('title')}")


if __name__ == "__main__":
    inspect_chunk_sizes()