import json
import math
import statistics
import time
from pathlib import Path

from src.rag.hybrid_retriever import HybridPolicyRetriever

EVAL_PATH = Path("../data/evaluation/gold_eval_v1.jsonl")


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 1.0
    top_k = set(retrieved[:k])
    return len(top_k.intersection(gold)) / len(gold)


def reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for idx, ref in enumerate(retrieved, start=1):
        if ref in gold_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int = 5) -> float:
    gold_set = set(gold)
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, ref in enumerate(retrieved[:k], start=1)
        if ref in gold_set
    )
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def run_benchmark(eval_path: Path = EVAL_PATH) -> None:
    retriever = HybridPolicyRetriever()
    rows = [json.loads(line) for line in eval_path.open("r", encoding="utf-8") if line.strip()]

    r3, r5, r10, mrrs, ndcgs, latencies_ms = [], [], [], [], [], []
    superseded_leaks = 0
    gate_correct, gate_total = 0, 0
    misses = []

    for row in rows:
        f = row.get("filters", {})
        t0 = time.perf_counter()
        out = retriever.search_policy_corpus(
            query=row["query"],
            region=f.get("region"),
            expense_category=f.get("expense_category"),
            grade=f.get("grade"),
            top_k=10,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        # Out-of-policy gate check
        if row.get("expect_no_evidence", False):
            gate_total += 1
            if out["status"] == "NO_POLICY_EVIDENCE":
                gate_correct += 1
            else:
                misses.append((row["eval_id"], "GATE_FALSE_POSITIVE", out["results"][0]["policy_ref"]))
            continue

        retrieved_refs = [hit["policy_ref"] for hit in out["results"]]
        if any(ref.startswith("SUP-") for ref in retrieved_refs):
            superseded_leaks += 1

        gold = row["gold_policy_refs"]
        rec5 = recall_at_k(retrieved_refs, gold, 5)
        r3.append(recall_at_k(retrieved_refs, gold, 3))
        r5.append(rec5)
        r10.append(recall_at_k(retrieved_refs, gold, 10))
        mrrs.append(reciprocal_rank(retrieved_refs, gold))
        ndcgs.append(ndcg_at_k(retrieved_refs, gold, 5))

        if rec5 < 1.0:
            misses.append((row["eval_id"], "RETRIEVAL_MISS", {"gold": gold, "got": retrieved_refs[:5]}))

    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0

    print("=== RAG PIPELINE BENCHMARK REPORT ===")
    print(f"In-Policy Queries : {len(r5)}")
    print(f"Recall@3          : {statistics.mean(r3):.3f}")
    print(f"Recall@5          : {statistics.mean(r5):.3f}")
    print(f"Recall@10         : {statistics.mean(r10):.3f}")
    print(f"MRR               : {statistics.mean(mrrs):.3f}")
    print(f"NDCG@5            : {statistics.mean(ndcgs):.3f}")
    print(f"Superseded Leaks  : {superseded_leaks}")
    if gate_total:
        print(f"Low-Conf Gate Acc : {gate_correct}/{gate_total} ({gate_correct/gate_total:.1%})")
    print(f"Latency p50 / p95 : {p50:.1f} ms / {p95:.1f} ms")
    print(f"\nFailures ({len(misses)}):")
    for m in misses[:10]:
        print(" ", m)


if __name__ == "__main__":
    run_benchmark()