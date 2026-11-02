// All backend communication lives here. The base URL comes from VITE_API_BASE_URL in .env.
import { BASE_URL, USE_MOCK } from "./config.js";
import { CancelledError, HttpError } from "./errors.js";
import * as demo from "./demoApi.js";

const TIMEOUT_MS = 120000; // the model can be slow, so allow 2 minutes

async function request(path, options = {}, signal) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, TIMEOUT_MS);
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", () => controller.abort(), { once: true });
  }
  try {
    const res = await fetch(`${BASE_URL}${path}`, { ...options, signal: controller.signal });
    if (!res.ok) {
      let detail = "";
      try {
        const body = await res.json();
        detail = body.detail || body.error || body.message || "";
        if (typeof detail !== "string") detail = JSON.stringify(detail);
      } catch {
        /* body was not JSON */
      }
      throw new HttpError(res.status, detail || `Server error ${res.status}`);
    }
    return await res.json().catch(() => ({}));
  } catch (err) {
    if (err.name === "AbortError") {
      if (timedOut) throw new Error("Request timed out. The model is too slow, please try again later.");
      if (signal?.aborted) throw new CancelledError();
    }
    if (err instanceof TypeError) {
      throw new Error(`Could not connect to the backend (${BASE_URL}). Is the server running?`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

// Maps the route to one of: none | single | multi | unknown
export function normalizeRoute(raw) {
  const r = String(raw ?? "").toLowerCase();
  if (!r) return "unknown";
  if (r.includes("multi")) return "multi";
  if (r.includes("single")) return "single";
  if (/^(no|none|direct|llm)([^a-z]|$)/.test(r)) return "none";
  return "unknown";
}

// Confidence: null/undefined/NaN -> null. 0..1 is converted to 0..100.
export function normalizeConfidence(raw) {
  if (raw === null || raw === undefined || raw === "") return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  const pct = n <= 1 ? n * 100 : n;
  return Math.max(0, Math.min(100, pct));
}

function normalizeChunk(c, i) {
  if (typeof c === "string") return { id: i, text: c, source: null, page: null, score: null };
  const text = c?.text ?? c?.content ?? c?.page_content ?? "";
  const source = c?.source ?? c?.metadata?.source ?? c?.doc ?? null;
  const rawPage = c?.page ?? c?.page_number ?? c?.metadata?.page ?? c?.metadata?.page_number;
  const page = rawPage !== null && rawPage !== undefined && Number.isFinite(Number(rawPage)) ? Number(rawPage) : null;
  const score = c?.score !== null && c?.score !== undefined && Number.isFinite(Number(c.score)) ? Number(c.score) : null;
  return { id: c?.id ?? i, text: String(text), source, page, score };
}

// Tolerates different key names in the backend response without crashing.
export function normalizeAnswer(raw) {
  const data = raw || {};
  const chunksRaw = data.chunks ?? data.retrieved_chunks ?? data.contexts ?? data.documents ?? [];
  const t = data.time_ms ?? data.latency_ms ?? data.elapsed_ms;
  return {
    answer: data.answer ?? data.response ?? "",
    route: normalizeRoute(data.route ?? data.strategy),
    confidence: normalizeConfidence(data.confidence),
    chunks: Array.isArray(chunksRaw) ? chunksRaw.map(normalizeChunk) : [],
    timeMs: t !== null && t !== undefined && Number.isFinite(Number(t)) ? Number(t) : null,
    stages: Array.isArray(data.stages) ? data.stages : [],
  };
}

export async function askQuestion(question, { signal } = {}) {
  const t0 = performance.now();
  const raw = USE_MOCK
    ? await demo.ask(question, signal)
    : await request(
        "/ask",
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) },
        signal
      );
  const result = normalizeAnswer(raw);
  result.clientMs = Math.round(performance.now() - t0);
  return result;
}

export async function uploadDocument(file) {
  let raw;
  if (USE_MOCK) {
    raw = await demo.upload(file);
  } else {
    const form = new FormData();
    form.append("file", file);
    raw = await request("/docs", { method: "POST", body: form });
  }
  const n = raw?.chunks_added ?? raw?.chunks;
  return {
    filename: raw?.filename ?? raw?.name ?? file.name,
    chunksAdded: n !== null && n !== undefined && Number.isFinite(Number(n)) ? Number(n) : null,
  };
}

// GET /docs (optional). If missing, the UI falls back to a locally remembered list.
export async function listDocuments() {
  if (USE_MOCK) return demo.list();
  const raw = await request("/docs");
  const arr = Array.isArray(raw) ? raw : raw?.documents ?? raw?.files ?? [];
  return arr
    .map((d) => (typeof d === "string" ? { name: d, chunks: null } : { name: d?.name ?? d?.filename ?? d?.source ?? "", chunks: d?.chunks ?? null }))
    .filter((d) => d.name);
}

// DELETE /docs/{name} (optional)
export async function deleteDocument(name) {
  if (USE_MOCK) return demo.remove(name);
  return request(`/docs/${encodeURIComponent(name)}`, { method: "DELETE" });
}
