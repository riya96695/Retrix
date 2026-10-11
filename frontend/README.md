# Frontend

Goal: a simple UI where a user types a query and sees the chosen route. Work on branch `frontend`.

Talks to the backend through `POST /route` (see `backend/README.md` for the exact request and response).

Suggested screens:
1. Query box + submit.
2. Result card: route (`NO_RETRIEVAL` / `SINGLE_STEP` / `MULTI_STEP`), probabilities, the LLM's hypothetical answer and its confidence.

Build against a mocked response first; switch to the backend URL once it is up.
