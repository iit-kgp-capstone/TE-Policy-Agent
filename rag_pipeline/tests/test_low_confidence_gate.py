import pytest
from src.rag.hybrid_retriever import HybridPolicyRetriever


@pytest.fixture(scope="module")
def retriever() -> HybridPolicyRetriever:
    # Replace 0.50 with the exact threshold output by calibrate_confidence_gate.py
    return HybridPolicyRetriever(confidence_threshold=0.50)


def test_in_domain_mumbai_passes_gate(retriever):
    res = retriever.search_policy_corpus(
        query="Mumbai hotel cap for L3 employee in January",
        region="IN",
        expense_category="hotel",
        grade="L3",
        top_k=3,
    )
    assert res["status"] == "OK"
    assert len(res["results"]) > 0
    assert "Mumbai" in res["results"][0]["title"]


@pytest.mark.parametrize("query", [
    "Hotel cap for Thimphu Bhutan for L3 employee",
    "Daily meal per diem for Antarctica research trip",
    "Reimbursement policy for private yacht charter",
])
def test_unlisted_queries_trigger_no_policy_evidence(retriever, query):
    res = retriever.search_policy_corpus(query=query, top_k=3)
    assert res["status"] == "NO_POLICY_EVIDENCE"
    assert res["trigger_decision"] == "HUMAN_REVIEW"
    assert res["results"] == []