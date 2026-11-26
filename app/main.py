import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from openai import APIError

from rag_pipeline.src.rag.hybrid_retriever import HybridPolicyRetriever
from .retrieval_agent import EvidenceVerificationError, PolicyRetrievalAgent, RetrievalRequest, RetrievalResponse

logger = logging.getLogger("uvicorn.error.policy_retrieval")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = PolicyRetrievalAgent(HybridPolicyRetriever())
    yield


app = FastAPI(title="T&E Policy Agent", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/api/v1/retrieve", response_model=RetrievalResponse)
def retrieve_policy(payload: RetrievalRequest, request: Request) -> RetrievalResponse:
    try:
        return request.app.state.agent.run(payload)
    except EvidenceVerificationError as exc:
        logger.warning("Policy evidence verification failed: %s", exc)
        raise HTTPException(502, "Retrieved policy evidence could not be verified.") from None
    except APIError as exc:
        logger.warning("Query embedding failed (%s)", type(exc).__name__)
        raise HTTPException(503, "The policy embedding service is unavailable.") from exc
