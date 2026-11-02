// Backend se baat karne wali saari logic yahin hai.
// Base URL .env ke VITE_API_BASE_URL se aata hai.
const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

const TIMEOUT_MS = 120000; // model slow ho sakta hai, isliye 2 minute

async function request(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE_URL}${path}`, { ...options, signal: controller.signal });
    if (!res.ok) {
      let detail = "";
      try {
        const body = await res.json();
        detail = body.detail || body.error || body.message || "";
        if (typeof detail !== "string") detail = JSON.stringify(detail);
      } catch {
        /* body JSON nahi tha */
      }
      throw new Error(detail || `Server error ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("Request timeout ho gaya. Model bahut slow hai, thodi der baad try karo.");
    }
    if (err instanceof TypeError) {
      throw new Error(`Backend se connect nahi hua (${BASE_URL}). Server chal raha hai?`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

// Route ko 3 fixed values me badalta hai: none | single | multi | unknown
export function normalizeRoute(raw) {
  const r = String(raw ?? "").toLowerCase();
  if (!r) return "unknown";
  if (r.includes("multi")) return "multi";
  if (r.includes("single")) return "single";
  if (r.includes("no") || r.includes("none") || r.includes("direct") || r.includes("llm")) return "none";
  return "unknown";
}

// Confidence: null/undefined/NaN -> null. 0..1 ko 0..100 me badalta hai.
export function normalizeConfidence(raw) {
  if (raw === null || raw === undefined || raw === "") return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  const pct = n <= 1 ? n * 100 : n;
  return Math.max(0, Math.min(100, pct));
}

function normalizeChunk(c, i) {
  if (typeof c === "string") return { id: i, text: c, source: null, score: null };
  const text = c?.text ?? c?.content ?? c?.page_content ?? "";
  const source = c?.source ?? c?.metadata?.source ?? c?.doc ?? null;
  const score = Number.isFinite(Number(c?.score)) ? Number(c.score) : null;
  return { id: c?.id ?? i, text: String(text), source, score };
}

// Backend ka response kuch alag key names me aaye to bhi crash na ho.
export function normalizeAnswer(raw) {
  const data = raw || {};
  const chunksRaw = data.chunks ?? data.retrieved_chunks ?? data.contexts ?? data.documents ?? [];
  return {
    answer: data.answer ?? data.response ?? "",
    route: normalizeRoute(data.route ?? data.strategy),
    confidence: normalizeConfidence(data.confidence),
    chunks: Array.isArray(chunksRaw) ? chunksRaw.map(normalizeChunk) : [],
  };
}

export async function askQuestion(question) {
  const raw = await request("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return normalizeAnswer(raw);
}

export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/docs", { method: "POST", body: form });
}
