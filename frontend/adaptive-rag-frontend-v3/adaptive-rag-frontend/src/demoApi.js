// DEMO MODE: sample data without a backend. Used only when VITE_USE_MOCK=true.
import { CancelledError } from "./errors.js";

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new CancelledError());
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        reject(new CancelledError());
      },
      { once: true }
    );
  });
}

let docs = [{ name: "sample.pdf", chunks: 12 }];

const CH = {
  bm25: { text: "BM25 is a keyword-based ranking algorithm. It uses term frequency and inverse document frequency to score how well a document matches a query. It is fast and works well on small datasets.", source: "sample.pdf", page: 2, score: 4.21 },
  vector: { text: "Vector search uses embeddings to find documents by meaning. It is more flexible than keyword search, but it needs an embedding model and a vector database.", source: "sample.pdf", page: 3, score: 3.4 },
  small: { text: "For small datasets BM25 is fast and works well, while vector search is better when the wording of the question differs from the wording of the documents.", source: "notes.txt", page: null, score: 1.9 },
  conf: { text: "The confidence score shows how many of the important query words were found in the retrieved chunk. When no search is performed, the confidence is null.", source: "sample.pdf", page: 4, score: 3.9 },
};

const stg = (llm, skip) => [
  { id: "query", ms: 4 },
  { id: "embedding", ms: 120, skipped: skip },
  { id: "retrieval", ms: 340, skipped: skip },
  { id: "rerank", ms: 210, skipped: skip },
  { id: "context", ms: 15, skipped: skip },
  { id: "llm", ms: llm },
  { id: "answer", ms: 3 },
];

export async function ask(question, signal) {
  const q = question.toLowerCase();
  if (/^\s*(hi|hii|hello|hey|thanks|thank you)\b/.test(q)) {
    await sleep(1500, signal);
    return { answer: "Hello! Ask me anything about your documents.", route: "no_search", confidence: null, chunks: [], time_ms: 1500, stages: stg(1400, true) };
  }
  if (/compare|\bvs\b|versus|difference/.test(q)) {
    await sleep(6000, signal);
    return {
      answer: "BM25 matches keywords, while vector search finds documents by meaning. BM25 is fast and simple on small datasets, whereas vector search is more flexible but needs an embedding model.",
      route: "multi_step", confidence: 0.64, chunks: [CH.bm25, CH.vector, CH.small], time_ms: 6000, stages: stg(4300, false),
    };
  }
  if (q.includes("bm25")) {
    await sleep(4200, signal);
    return { answer: "BM25 is a keyword-based ranking algorithm. It uses term frequency and inverse document frequency, and it is fast on small datasets.", route: "single_search", confidence: 0.87, chunks: [CH.bm25], time_ms: 4200, stages: stg(3100, false) };
  }
  if (q.includes("confidence")) {
    await sleep(4000, signal);
    return { answer: "Confidence is null when no search is performed (the No search route). When a search happens, it shows how many of the important query words were found in the chunk.", route: "single_search", confidence: 0.52, chunks: [CH.conf], time_ms: 4000, stages: stg(2900, false) };
  }
  await sleep(3500, signal);
  return { answer: "I could not find an answer to this question in the documents.", route: "single_search", confidence: 0.12, chunks: [], time_ms: 3500, stages: stg(2400, false) };
}

export async function upload(file, signal) {
  await sleep(3500, signal);
  const chunks = Math.max(1, Math.round(file.size / 500));
  docs = [{ name: file.name, chunks }, ...docs.filter((d) => d.name !== file.name)];
  return { filename: file.name, chunks_added: chunks };
}

export async function list() {
  await sleep(200);
  return docs.map((d) => ({ ...d }));
}

export async function remove(name) {
  await sleep(300);
  docs = docs.filter((d) => d.name !== name);
  return { ok: true };
}
