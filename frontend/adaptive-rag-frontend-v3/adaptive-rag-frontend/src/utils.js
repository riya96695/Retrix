import { QUERY_STAGES } from "./config.js";

export function loadJSON(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v ? JSON.parse(v) : fallback;
  } catch {
    return fallback;
  }
}

export function saveJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage is full or blocked */
  }
}

export function formatMs(ms) {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return null;
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

// active = step running right now (-1 = none). Earlier steps are "done", later ones "pending".
export function statusesFromActive(count, active, activeStatus = "active") {
  return Array.from({ length: count }, (_, i) =>
    active < 0 ? "pending" : i < active ? "done" : i === active ? activeStatus : "pending"
  );
}

export const hasBackendStages = (result) => Array.isArray(result.stages) && result.stages.length > 0;

// Uses the backend's stages if present, otherwise estimates from the route.
export function buildStageMeta(result) {
  const skipDefault = result.route === "none" ? new Set(["embedding", "retrieval", "rerank", "context"]) : new Set();
  const byId = {};
  (result.stages || []).forEach((s) => {
    const n = String(s?.id ?? s?.name ?? s?.stage ?? "").toLowerCase();
    const found = QUERY_STAGES.find((st) => n.includes(st.id.slice(0, 5)));
    if (found) byId[found.id] = s;
  });
  return QUERY_STAGES.map((st) => {
    const b = byId[st.id];
    const skipped = b ? b.skipped === true || b.status === "skipped" : skipDefault.has(st.id);
    const ms = b && b.ms !== null && b.ms !== undefined && Number.isFinite(Number(b.ms)) ? Math.round(Number(b.ms)) : null;
    const detail = skipped ? "skipped" : b?.detail ? String(b.detail) : ms !== null ? formatMs(ms) : null;
    return { status: skipped ? "skipped" : "done", detail };
  });
}

// "sample.pdf (page 2), notes.txt"
export function collectSources(chunks) {
  const map = new Map();
  (chunks || []).forEach((c) => {
    if (!c.source) return;
    if (!map.has(c.source)) map.set(c.source, new Set());
    if (c.page !== null && c.page !== undefined) map.get(c.source).add(c.page);
  });
  return [...map].map(([name, pages]) => {
    const p = [...pages].sort((a, b) => a - b);
    return p.length ? `${name} (page ${p.join(", ")})` : name;
  });
}

const SKIP_WORDS = new Set(["the", "and", "for", "with", "how", "what", "when", "where", "why", "who", "are", "does", "is", "this", "that", "kya", "hai", "aur"]);

export function queryTerms(q) {
  const words = String(q || "").toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) || [];
  return [...new Set(words)].filter((w) => !SKIP_WORDS.has(w));
}

export function splitHighlight(text, terms) {
  if (!terms.length) return [{ t: text, hit: false }];
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp("(" + escaped.join("|") + ")", "gi");
  return text
    .split(re)
    .filter(Boolean)
    .map((part) => ({ t: part, hit: terms.includes(part.toLowerCase()) }));
}
