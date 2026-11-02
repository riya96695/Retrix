# Adaptive RAG Frontend (v3)

    npm install
    copy .env.example .env      # Mac/Linux: cp .env.example .env
    npm run dev

In .env:
  VITE_USE_MOCK=true   -> demo mode (works without a backend)
  VITE_USE_MOCK=false  -> real backend (set VITE_API_BASE_URL)
Restart `npm run dev` after changing .env.

What the backend must provide: BACKEND_SPEC.md
Example questions can be changed in src/config.js
