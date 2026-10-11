# Backend (FastAPI)

Goal: expose the router as an HTTP API. Work on branch `backend`.

## Contract (agreed with core and frontend)

`POST /route`

Request:
```json
{"query": "Who wrote Hamlet?"}
```

Response:
```json
{
  "route": "NO_RETRIEVAL | SINGLE_STEP | MULTI_STEP",
  "probs": {"NO_RETRIEVAL": 0.6, "SINGLE_STEP": 0.3, "MULTI_STEP": 0.1},
  "hypothetical_answer": "...",
  "confidence": {"mean_lp": -0.3, "min_lp": -1.2, "first_lp": -0.1, "first_ent": 0.8, "n_tok": 4}
}
```

## Plan

1. Start with a **stub** that returns a hard-coded response in this exact shape, so the frontend can be built in parallel.
2. Later replace the stub with the real `route(query)` from `src/` (to be added on branch `core`).

## Notes

- The real router needs the Qwen2.5-3B LLM (GPU). Develop against the stub on a laptop.
- Confidence features are specific to Qwen2.5-3B; another LLM would need the router retrained.
- Retrieval over user documents (chunking + index) is not built yet. `src/bm25.py` can index a list of `{doc_id, title, text}`.
