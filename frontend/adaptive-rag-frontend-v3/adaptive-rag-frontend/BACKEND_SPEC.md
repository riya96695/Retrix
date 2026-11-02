# Backend spec (what the frontend expects)

## POST /ask  (required)
Request:  { "question": "..." }
Response:
{
  "answer": "...",
  "route": "no_search" | "single_search" | "multi_step",
  "confidence": 0.82,              // 0..1 or 0..100, null when no search was done
  "chunks": [ { "text": "...", "source": "sample.pdf", "page": 2, "score": 0.91 } ],

  // Optional (if missing, the frontend estimates them):
  "time_ms": 3200,                 // total time taken
  "stages": [                      // real stage-by-stage info
    { "id": "query",     "ms": 4 },
    { "id": "embedding", "ms": 120 },
    { "id": "retrieval", "ms": 340, "detail": "4 chunks" },
    { "id": "rerank",    "ms": 210 },
    { "id": "context",   "ms": 15 },
    { "id": "llm",       "ms": 2500 },
    { "id": "answer",    "ms": 3 }
  ]                                // a skipped stage: "skipped": true
}

## POST /docs  (optional)  multipart/form-data, field "file"
Response: { "filename": "sample.pdf", "chunks_added": 12 }

## GET /docs  (optional)  list of uploaded files
Response: { "documents": ["sample.pdf", "notes.txt"] }
(or objects: [{ "name": "sample.pdf", "chunks": 12 }])
If missing, the frontend shows a list remembered in the browser.

## DELETE /docs/{filename}  (optional)
If missing, "Remove" only removes the file from the list (the UI says so).

## Also
- Allow the frontend origin in CORS (localhost:5173 and your deploy URL).
- All responses must be JSON.
