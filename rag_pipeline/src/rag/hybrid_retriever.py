import re
from itertools import zip_longest
from pathlib import Path
from typing import Any

import chromadb
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi

from .ingest_chroma import CHROMA_DIR, COLLECTION_NAME, HostedEmbedder

GRADE_TO_BAND = {
    "L1": "L1-L4", "L2": "L1-L4", "L3": "L1-L4", "L4": "L1-L4",
    "L5": "L5-L6", "L6": "L5-L6",
    "L7": "L7-L8", "L8": "L7-L8",
}


def _tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer preserving policy IDs (POL-IN-HOT-0046) and grades (L4)."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class HybridPolicyRetriever:
    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        reranker_model: str = "ms-marco-MiniLM-L-12-v2",
        confidence_threshold: float = 0.25,
    ):
        if not chroma_dir.is_dir():
            raise ValueError("Policy index is missing. Run corpus ingestion first.")
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_collection(collection_name, embedding_function=None)
        self.confidence_threshold = confidence_threshold

        # Load all chunks from ChromaDB once so BM25 and Dense share identical IDs & metadata
        raw = self.collection.get(include=["documents", "metadatas"])
        if not raw["ids"]:
            raise ValueError("Policy index is empty. Run corpus ingestion first.")
        self.embedder = HostedEmbedder(cache_path=None)
        self.ranker = Ranker(model_name=reranker_model)
        self.ids: list[str] = raw["ids"]
        self.docs: list[str] = raw["documents"]
        self.metas: list[dict[str, Any]] = raw["metadatas"]
        self.hotel_titles_by_city: dict[str, set[str]] = {}
        for meta in self.metas:
            if meta.get("expense_category") != "hotel" or meta.get("status") != "ACTIVE":
                continue
            match = re.match(r"^(.+?) \(Tier-[^)]+\) Hotel Cap\b", meta["title"])
            if match:
                self.hotel_titles_by_city.setdefault(match[1].casefold(), set()).add(meta["title"])
        self.id_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(self.ids)}

        self.tokenized_corpus = [_tokenize(doc) for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _build_chroma_where(
        self,
        region: str | None,
        expense_category: str | None,
        grade: str | None,
        include_superseded: bool,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if not include_superseded:
            clauses.append({"status": {"$eq": "ACTIVE"}})
        if region:
            clauses.append({"region": {"$in": [region.upper(), "GLOBAL"]}})
        if expense_category:
            clauses.append({"expense_category": {"$eq": expense_category.lower()}})
        if grade and grade.upper() in GRADE_TO_BAND:
            band = GRADE_TO_BAND[grade.upper()]
            clauses.append({"grade_band": {"$in": [band, "ALL"]}})

        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _matches_meta(
        self,
        meta: dict[str, Any],
        region: str | None,
        expense_category: str | None,
        grade: str | None,
        include_superseded: bool,
    ) -> bool:
        if not include_superseded and meta.get("status") != "ACTIVE":
            return False
        if region and meta.get("region") not in (region.upper(), "GLOBAL"):
            return False
        if expense_category and meta.get("expense_category") != expense_category.lower():
            return False
        if grade and grade.upper() in GRADE_TO_BAND:
            band = GRADE_TO_BAND[grade.upper()]
            if meta.get("grade_band") not in (band, "ALL"):
                return False
        return True

    def _bm25_search(
        self,
        query: str,
        region: str | None,
        expense_category: str | None,
        grade: str | None,
        include_superseded: bool,
        top_n: int = 25,
        titles: set[str] | None = None,
    ) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        scored = [
            (self.ids[i], float(scores[i]))
            for i in range(len(self.ids))
            if scores[i] > 0
            and (titles is None or self.metas[i]["title"] in titles)
            and self._matches_meta(self.metas[i], region, expense_category, grade, include_superseded)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scored[:top_n]]

    def _dense_search(
        self,
        query_vector: list[float],
        where: dict[str, Any] | None,
        top_n: int = 25,
    ) -> list[str]:
        res = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_n,
            where=where,
        )
        return res["ids"][0] if res["ids"] else []

    @staticmethod
    def _rrf_fuse(bm25_ids: list[str], dense_ids: list[str], k: int = 60, top_n: int = 15) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for rank, cid in enumerate(bm25_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        for rank, cid in enumerate(dense_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:top_n]

    def search_policy_corpus(
        self,
        query: str,
        region: str | None = None,
        expense_category: str | None = None,
        grade: str | None = None,
        include_superseded: bool = False,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Deterministic Hybrid RAG tool exposed to the LangGraph Executor.
        Returns grounded chunks with exact policy_ref IDs and reranker scores.
        """
        where = self._build_chroma_where(region, expense_category, grade, include_superseded)

        matched_cities = []
        is_hotel_search = (
            expense_category.lower() == "hotel"
            if expense_category is not None
            else re.search(r"\bhotels?\b", query, re.I) is not None
        )
        if not include_superseded and is_hotel_search:
            matched_cities = [
                city for city in self.hotel_titles_by_city
                if re.search(rf"(?<!\w){re.escape(city)}(?!\w)", query.casefold())
            ]
        query_vector = self.embedder.embed_texts([query])[0]
        if not matched_cities:
            return self._search(query, query_vector, where, region, expense_category, grade, include_superseded, top_k)

        city_searches = {}
        for city in matched_cities:
            titles = self.hotel_titles_by_city[city]
            title_filter = {"title": {"$in": sorted(titles)}}
            city_filter = {"$and": [where, title_filter]} if where else title_filter
            city_searches[city] = self._search(
                query, query_vector, city_filter, region, expense_category, grade,
                include_superseded, top_k, titles,
            )

        if len(city_searches) == 1:
            return next(iter(city_searches.values()))

        results = [
            excerpt
            for batch in zip_longest(*(search["results"] for search in city_searches.values()))
            for excerpt in batch
            if excerpt is not None
        ][:top_k]
        return {
            "status": "OK" if results else "NO_POLICY_EVIDENCE",
            "results": results,
            "debug_trace": {city: search["debug_trace"] for city, search in city_searches.items()},
        }

    def _search(
        self,
        query: str,
        query_vector: list[float],
        where: dict[str, Any] | None,
        region: str | None,
        expense_category: str | None,
        grade: str | None,
        include_superseded: bool,
        top_k: int,
        titles: set[str] | None = None,
    ) -> dict[str, Any]:

        bm25_ids = self._bm25_search(
            query, region, expense_category, grade, include_superseded, top_n=25, titles=titles,
        )
        dense_ids = self._dense_search(query_vector, where, top_n=25)
        fused = self._rrf_fuse(bm25_ids, dense_ids, k=60, top_n=15)

        if not fused:
            return {
                "status": "NO_POLICY_EVIDENCE",
                "results": [],
                "debug_trace": {"bm25_count": 0, "dense_count": 0, "fused_count": 0},
            }

        passages = [
            {
                "id": cid,
                "text": self.docs[self.id_to_idx[cid]],
                "meta": self.metas[self.id_to_idx[cid]],
            }
            for cid, _ in fused
        ]

        reranked = self.ranker.rerank(RerankRequest(query=query, passages=passages))
        top_hits = reranked[:top_k]

        top_score = float(top_hits[0]["score"]) if top_hits else 0.0
        if top_score < self.confidence_threshold:
            return {
                "status": "NO_POLICY_EVIDENCE",
                "top_score": round(top_score, 4),
                "results": [],
                "debug_trace": {"bm25_top5": bm25_ids[:5], "dense_top5": dense_ids[:5]},
            }

        formatted = [
            {
                "chunk_id": hit["id"],
                "policy_ref": hit["meta"]["policy_ref"],
                "title": hit["meta"]["title"],
                "region": hit["meta"]["region"],
                "grade_band": hit["meta"]["grade_band"],
                "expense_category": hit["meta"]["expense_category"],
                "effective_date": hit["meta"]["effective_date"],
                "rerank_score": round(float(hit["score"]), 4),
                "text": hit["text"],
            }
            for hit in top_hits
        ]

        return {
            "status": "OK",
            "results": formatted,
            "debug_trace": {
                "bm25_top5": bm25_ids[:5],
                "dense_top5": dense_ids[:5],
                "fused_count": len(fused),
            },
        }


if __name__ == "__main__":
    retriever = HybridPolicyRetriever()
    out = retriever.search_policy_corpus(
        query="Mumbai hotel cap for L3 employee in January",
        region="IN",
        expense_category="hotel",
        grade="L3",
        top_k=3,
    )
    print("Status:", out["status"])
    for r in out["results"]:
        print(f"  [{r['rerank_score']:.4f}] {r['policy_ref']} | {r['title']}")
