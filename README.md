# Retrix: Adaptive RAG with Confidence-Calibrated Routing

Decide **how much retrieval a question needs before retrieving**: none, one search, or iterative search.

## Status

| Part | State |
|---|---|
| Data (1200 queries, 6437-paragraph corpus), BM25 retriever, outcome-based labels | done |
| Query router experiments (frozen encoder, LLM confidence features) | done |
| Saved final router (`models/router.npz`) + `route(query)` (`src/route.py`) | done (router tested on Kaggle) |
| Document store + routed RAG (`src/rag.py`: chunking, BM25, no / one / iterative search) | written, not yet tested on a GPU |
| Confidence calibration (conformal), answer / retry / refuse decision, RL router | not done |
| End-to-end evaluation (answer accuracy vs. number of searches), `src/eval_end2end.py` | script written, needs `--all` runs of B and C |
| Unseen test set (2Wiki / MuSiQue) | not done |
| Backend (FastAPI) and frontend | in progress (see `backend/`, `frontend/`) |

## Idea

1. The LLM answers the query **without search**; we also record the LLM's own confidence (token log-probabilities).
2. A small router predicts one of `NO_RETRIEVAL`, `SINGLE_STEP`, `MULTI_STEP`.
3. **Labels come from outcomes, not dataset names.** For every query we try the cheapest strategy first and keep the first one that answers correctly:
   - A: LLM alone, no search -> `NO_RETRIEVAL`
   - B: one BM25 search + LLM -> `SINGLE_STEP`
   - C: iterative BM25 search (LLM writes follow-up searches, max 3 rounds) + LLM -> `MULTI_STEP`
   - none correct -> *fallback* label from the dataset type (reported separately and excluded from router training)

## Built from scratch

BM25 retriever, MLP router (manual backprop), Adam optimiser, class-weighted loss, metrics and cross-validation are written from scratch (numpy).
Pretrained models are used as **frozen components**: `Qwen/Qwen2.5-3B-Instruct` (LLM) and `BAAI/bge-small-en-v1.5` (encoder).
The `datasets` library is used only to download SQuAD and HotpotQA.

## Repository layout

```
src/         core: data, retrieval, labels, router experiments, route.py, rag.py
models/      saved router (router.npz)
data/        generated CSVs (queries, answers_A/B/C, labels, conf_A) and corpus.jsonl
backend/     FastAPI service (separate branch)
frontend/    UI (separate branch)
```

## Reproduce

```
pip install -r requirements.txt
python src/data.py             # 600 queries per dataset + corpus
python src/gen_answers.py      # strategy A   (GPU)
python src/gen_answers_B.py    # strategy B   (GPU)
python src/gen_answers_C.py    # strategy C   (GPU)
python src/make_labels.py      # strict and relaxed labels
python src/gen_conf.py         # LLM confidence features (GPU)
python src/embed.py            # frozen bge-small embeddings
python src/train_classifier.py # 5-fold CV x 5 seeds
python src/train_final.py      # saves models/router.npz (confidence-only router)

# end-to-end evaluation (B and C on every query, GPU), then the table
python src/gen_answers_B.py --all
python src/gen_answers_C.py --all
python src/eval_end2end.py
```

### Try the router and the RAG (needs a GPU, e.g. Kaggle)

```python
import sys; sys.path.insert(0, "src")
from rag import RagSystem
from route import Retrix

rag = RagSystem(Retrix())
rag.store.add_text("notes", "Paris is the capital of France. ...")
print(rag.ask("What is the capital of France?"))
```

The GPU steps were run on a Kaggle T4. All generation scripts are resumable (they checkpoint every 20 queries).

## Results so far

Labels (relaxed matcher, 1200 queries): `NO_RETRIEVAL` 201, `SINGLE_STEP` 656, `MULTI_STEP` 343.
Only the 832 outcome-labelled queries are used for the router (201 / 545 / 86); 368 are fallback.

Router, 5-fold CV x 5 seeds (mean +- std over seeds):

| Input | macro-F1 |
|---|---|
| dataset name only (always `SINGLE_STEP`) | 0.264 |
| query only | 0.396 +- 0.010 |
| query [SEP] answer | 0.375 +- 0.004 |
| [query ; answer] | 0.406 +- 0.009 |
| LLM confidence only | 0.436 +- 0.008 |
| [query ; answer ; confidence] | 0.440 +- 0.013 |

Main findings: the LLM's own confidence is the strongest routing signal (mostly for `NO_RETRIEVAL`, F1 0.32 -> 0.61);
appending the hypothetical answer to the query did not help with a frozen encoder;
`SINGLE_STEP` vs `MULTI_STEP` is not learnable from these inputs (MULTI F1 below 0.17).

Negative results (kept on purpose): taking 5 instead of 3 paragraphs per round, and expanding the second search with paragraph titles, both gave fewer correct multi-step answers.

## Limitations

- Labels depend on the LLM, on BM25 and on the matcher (strict vs relaxed differ for 22 labels; Qwen 1.5B vs 3B labels agree on ~81%).
- ~31% of queries end up as fallback; `MULTI_STEP` has only 86 outcome-labelled examples.
- The corpus is small (paragraphs of the sampled queries) and BM25 is keyword-based.
- No unseen-dataset test and no end-to-end evaluation yet; confidence features are specific to the Qwen2.5-3B model.
- Some SQuAD questions are ambiguous without their passage.

## Team workflow

Branches: `main` (stable), `core`, `backend`, `frontend`. Commit to your own branch and open a Pull Request into `main`.
