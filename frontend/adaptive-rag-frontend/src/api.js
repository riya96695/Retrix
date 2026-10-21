import { mockAnswer } from "./mock.js";

const USE_MOCK = import.meta.env.VITE_USE_MOCK !== "false";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function askQuery(query, { simulateError = false } = {}) {
  if (USE_MOCK) {
    await sleep(1400);
    if (simulateError) throw new Error("Could not reach the server (mock error).");
    return mockAnswer(query);
  }
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, language: "en" }),
  });
  if (!res.ok) throw new Error(`Server error: HTTP ${res.status}`);
  return res.json();
}

export async function uploadFile(file) {
  if (USE_MOCK) {
    await sleep(1200 + Math.random() * 800);
    return { chunks: Math.max(3, Math.round(file.size / 900)) };
  }
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: fd });
  if (!res.ok) throw new Error(`Upload failed: HTTP ${res.status}`);
  return res.json();
}
