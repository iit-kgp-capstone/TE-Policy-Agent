from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from rag_pipeline.src.rag.hybrid_retriever import HybridPolicyRetriever

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Filter = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_query: Query
    region: Filter | None = None
    expense_category: Filter | None = None
    grade: Annotated[
        str, StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"^[Ll][1-8]$")
    ] | None = None
    top_k: int = Field(default=5, ge=1, le=5)


class PolicyEvidence(BaseModel):
    chunk_id: str
    policy_ref: str
    title: str
    region: str
    grade_band: str
    expense_category: str
    effective_date: str
    rerank_score: float
    text: str


class RetrievalResponse(BaseModel):
    status: Literal["OK", "NO_POLICY_EVIDENCE"]
    results: list[PolicyEvidence]


class RetrievalState(TypedDict, total=False):
    request: RetrievalRequest
    response: RetrievalResponse


class EvidenceVerificationError(Exception):
    """The retrieval result could not be verified against the policy index."""


class PolicyRetrievalAgent:
    def __init__(self, retriever: HybridPolicyRetriever):
        self.retriever = retriever
        graph = StateGraph(RetrievalState)
        graph.add_node("retrieve_policy", self._retrieve_policy)
        graph.add_node("verify_evidence", self._verify_evidence)
        graph.add_edge(START, "retrieve_policy")
        graph.add_edge("retrieve_policy", "verify_evidence")
        graph.add_edge("verify_evidence", END)
        self.graph = graph.compile()

    def _retrieve_policy(self, state: RetrievalState) -> RetrievalState:
        request = state["request"]
        result = self.retriever.search_policy_corpus(
            query=request.user_query,
            region=request.region,
            expense_category=request.expense_category,
            grade=request.grade,
            top_k=request.top_k,
        )
        try:
            return {"response": RetrievalResponse.model_validate(result)}
        except ValidationError as exc:
            raise EvidenceVerificationError("Invalid retrieval response") from exc

    def _verify_evidence(self, state: RetrievalState) -> RetrievalState:
        request = state["request"]
        response = state["response"]
        chunk_ids = [excerpt.chunk_id for excerpt in response.results]
        if (
            (response.status == "OK") != bool(chunk_ids)
            or len(chunk_ids) > request.top_k
            or len(chunk_ids) != len(set(chunk_ids))
        ):
            raise EvidenceVerificationError("Inconsistent retrieval response")
        if not chunk_ids:
            return {}

        indexed = self.retriever.collection.get(
            ids=chunk_ids, include=["documents", "metadatas"],
        )
        by_id = dict(zip(indexed["ids"], zip(indexed["documents"], indexed["metadatas"])))
        for excerpt in response.results:
            text, metadata = by_id.get(excerpt.chunk_id, (None, None))
            expected = excerpt.model_dump(exclude={"chunk_id", "text", "rerank_score"})
            if (
                text != excerpt.text
                or not metadata
                or metadata.get("status") != "ACTIVE"
                or any(metadata.get(field) != value for field, value in expected.items())
            ):
                raise EvidenceVerificationError("Excerpt does not match an active indexed policy")
        return {}

    def run(self, request: RetrievalRequest) -> RetrievalResponse:
        return self.graph.invoke({"request": request})["response"]
