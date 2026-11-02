# Adaptive RAG Frontend

## Chalane ka tareeka
    npm install
    cp .env.example .env     # VITE_API_BASE_URL apne backend pe set karo
    npm run dev

## Backend se ye expected hai
POST /ask   body: { "question": "..." }
response:
    {
      "answer": "...",
      "route": "no_search" | "single_search" | "multi_step",
      "confidence": 0.82,            // ya null jab search nahi hua
      "chunks": [ { "text": "...", "source": "file.pdf", "score": 0.91 } ]
    }

POST /docs  multipart/form-data, field name: "file"

Backend me CORS allow karna mat bhoolna (localhost:5173).
