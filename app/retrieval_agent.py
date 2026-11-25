from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

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


class PolicyRetrievalAgent:
    def __init__(self, retriever: HybridPolicyRetriever):
        self.retriever = retriever
        graph = StateGraph(RetrievalState)
        graph.add_node("retrieve_policy", self._retrieve_policy)
        graph.add_edge(START, "retrieve_policy")
        graph.add_edge("retrieve_policy", END)
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
        return {"response": RetrievalResponse.model_validate(result)}

    def run(self, request: RetrievalRequest) -> RetrievalResponse:
        return self.graph.invoke({"request": request})["response"]
