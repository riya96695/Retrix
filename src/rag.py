"""RAG on top of the router.
Documents -> chunks -> BM25 index (scratch). The router decides how to answer:
  NO_RETRIEVAL -> the LLM's own answer, no search
  SINGLE_STEP  -> one BM25 search over the chunks, LLM answers from them      (same as strategy B)
  MULTI_STEP   -> iterative search, the LLM writes follow-up searches (max 3)  (same as strategy C)
SINGLE_STEP also returns answer_confidence (Step 2 score, 0..1); it is None on the other routes.

Needs a GPU for the LLM (run on Kaggle/Colab).
Usage:  python src/rag.py --doc notes.txt --doc more.txt "your question"
"""
import argparse
import json
import os

from bm25 import BM25
from route import Retrix

TOP_K_B, TOP_K_C, MAX_ROUNDS = 5, 3, 3   # same values as gen_answers_B.py / gen_answers_C.py


def chunk_words(text, size=120, overlap=30):
    """Split text into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), size - overlap)]


class DocStore:
    """User documents as chunks, searched with our own BM25."""

    def __init__(self, size=120, overlap=30):
        self.size, self.overlap = size, overlap
        self.chunks, self.index = [], None

    def add_text(self, name, text):
        new = chunk_words(text, self.size, self.overlap)
        for i, c in enumerate(new):
            self.chunks.append({"doc_id": f"{name}#{i}", "title": name, "text": c})
        self.index = BM25(self.chunks) if self.chunks else None
        return len(new)

    def add_file(self, path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return self.add_text(os.path.basename(path), f.read())

    def search(self, query, k=5):
        """Returns [(chunk, score)], best first."""
        if not self.index:
            return []
        by_id = {c["doc_id"]: c for c in self.chunks}
        return [(by_id[d], s) for d, s in self.index.search(query, k=k)]


def _context(chunks):
    return "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)


class RagSystem:
    def __init__(self, retrix, store=None):
        self.retrix, self.store = retrix, store or DocStore()
        try:                                     # Step 2 score; the system still works without it
            from live_signals import LiveConfidence
            self.conf = LiveConfidence(retrix.llm)
        except Exception as e:
            self.conf = None
            print(f"answer_confidence disabled: {e}")

    def ask(self, query):
        r = self.retrix.route(query)            # route, probs, hypothetical_answer, confidence
        llm = self.retrix.llm
        hits, searches, note, answer_conf = [], 0, "", None

        if r["route"] == "NO_RETRIEVAL":
            final = r["hypothetical_answer"]
        elif not self.store.chunks:
            final = r["hypothetical_answer"]
            note = "no documents loaded: returning the LLM's own answer"
        elif r["route"] == "SINGLE_STEP":
            hits, searches = self.store.search(query, TOP_K_B), 1
            if self.conf:
                final, answer_conf = self.conf.answer_with_score(query, hits)
            else:
                final = llm.generate(
                    "Use the context to answer. Answer in a few words, no explanation.\n\n"
                    f"Context:\n{_context([c for c, _ in hits])}\n\nQuestion: {query}")
        else:                                    # MULTI_STEP: LLM writes follow-up searches
            seen, search_q, final = {}, query, ""
            for rnd in range(1, MAX_ROUNDS + 1):
                searches += 1
                for c, s in self.store.search(search_q, TOP_K_C):
                    seen.setdefault(c["doc_id"], (c, s))
                last = rnd == MAX_ROUNDS
                if last:
                    instr = "Answer the question in a few words, no explanation."
                else:
                    instr = ("If the context is enough, reply exactly 'ANSWER: <short answer>'. "
                             "Otherwise reply exactly 'SEARCH: <short new search query for the missing fact>'.")
                reply = llm.generate(
                    f"{instr}\n\nContext:\n{_context([c for c, _ in seen.values()])}\n\nQuestion: {query}")
                if not last and reply.upper().startswith("SEARCH:"):
                    search_q = reply[7:].strip() or query
                    continue
                final = reply.split(":", 1)[1].strip() if reply.upper().startswith("ANSWER:") else reply
                break
            hits = list(seen.values())

        return {**r, "final_answer": final, "n_searches": searches, "note": note,
                "answer_confidence": answer_conf,
                "retrieved": [{"chunk_id": c["doc_id"], "doc": c["title"], "chunk": c["text"],
                               "score": round(float(s), 3)} for c, s in hits]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", action="append", default=[], help="path to a .txt file (repeatable)")
    ap.add_argument("query", nargs="+")
    args = ap.parse_args()

    system = RagSystem(Retrix())
    for path in args.doc:
        print(f"{path}: {system.store.add_file(path)} chunks")
    print(json.dumps(system.ask(" ".join(args.query)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()