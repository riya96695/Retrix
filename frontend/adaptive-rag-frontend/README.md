# Adaptive RAG Frontend (React + Vite)

## Run
    npm install
    npm run dev        # http://localhost:5173

## Mock vs real backend
`.env` mein `VITE_USE_MOCK=true` (default) -> mocked response.
Backend ready hone par `VITE_USE_MOCK=false` karein. Vite dev server `/api` ko `http://localhost:8000` par proxy karta hai.

## API contract
POST /api/query   body: {"query": "..."}
  -> { route, probabilities:{no_retrieval,single_step,multi_step}, llm_answer, llm_confidence,
       evaluator:{verdict:"correct|ambiguous|incorrect|not_used", score}, final_answer,
       sources:[{doc, chunk_id, score, text}] }
POST /api/upload  multipart field "file" -> { chunks: <int> }
