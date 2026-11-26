# TE-Policy-Agent

## Policy retrieval demo

A single LangGraph agent takes a human query, calls the existing hybrid
retriever, and returns ranked policy excerpts and references. The retriever
embeds the query, searches Chroma and BM25, and reranks the fused candidates.

Set `EMBEDDING_API_KEY` (or `OPENAI_API_KEY`) in your environment or a local
`.env`. For another OpenAI-compatible provider or model, also set
`EMBEDDING_BASE_URL` and `EMBEDDING_MODEL`. Use the same embedding
configuration for ingestion and queries.

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m rag_pipeline.src.rag.ingest_chroma
python -m uvicorn app.main:app --port 8001
```

Ingestion creates `data/chroma_db`. FlashRank may download its reranker
model on first startup.

Try a query in the [API docs](http://127.0.0.1:8001/docs) or in PowerShell.

```powershell
$body = @{
    user_query = "What is the hotel limit for an L3 employee in Mumbai in January?"
    region = "IN"
    expense_category = "hotel"
    grade = "L3"
    top_k = 3
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8001/api/v1/retrieve -Method Post -ContentType 'application/json' -Body $body
```

Only `user_query` is required. `OK` includes ranked excerpts, references,
and reranker scores. `NO_POLICY_EVIDENCE` means the retriever found no
evidence above its relevance threshold.

Hotel searches that name one city from the indexed hotel-cap titles are
scoped to that city's active hotel caps. The grade filter still applies.

After retrieval, the graph verifies each excerpt against its original Chroma
chunk. It checks the stored text, citation, policy metadata, and active status.
Duplicate or inconsistent results and mismatched evidence return HTTP 502.
This check covers source integrity. Claim amounts and approval decisions
remain outside the retrieval endpoint.

Run the API test with `python -m pytest tests/test_retrieval_api.py`. It
uses a temporary Chroma index and substitutes the embedding and reranking
calls so it can run offline.
