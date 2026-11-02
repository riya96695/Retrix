export const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
export const USE_MOCK = String(import.meta.env.VITE_USE_MOCK ?? "true").toLowerCase() !== "false";

// Example questions (one click to ask). Change them to match your documents.
export const EXAMPLE_QUESTIONS = [
  "Hello!",
  "What is BM25?",
  "Compare BM25 and vector search",
  "When is confidence null?",
];

// Query stages (lower half of your pipeline image)
export const QUERY_STAGES = [
  { id: "query", label: "Query", icon: "💬", color: "#6366f1", info: "Your question has reached the system." },
  { id: "embedding", label: "Embedding", icon: "🧬", color: "#8b5cf6", info: "The question is converted into a vector (a list of numbers) so it can be searched by meaning." },
  { id: "retrieval", label: "Retrieval", icon: "🔎", color: "#0ea5e9", info: "The most similar chunks are found in your documents." },
  { id: "rerank", label: "Rerank", icon: "📊", color: "#14b8a6", info: "The retrieved chunks are re-scored and the most useful ones are moved to the top." },
  { id: "context", label: "Context", icon: "🧩", color: "#f59e0b", info: "The best chunks and your question are combined into a prompt for the model." },
  { id: "llm", label: "LLM", icon: "🧠", color: "#ec4899", info: "The language model reads the prompt and writes the answer. A slow model spends most of its time here." },
  { id: "answer", label: "Answer", icon: "✨", color: "#22c55e", info: "The answer is ready, together with its sources." },
];

// Indexing stages (upper half of your pipeline image)
export const INDEX_STAGES = [
  { id: "documents", label: "Documents", icon: "📄", color: "#3b82f6", info: "The file has reached the server." },
  { id: "parsing", label: "Parsing", icon: "⚙️", color: "#8b5cf6", info: "Text and structure are extracted from the file." },
  { id: "chunking", label: "Chunking", icon: "✂️", color: "#f97316", info: "The text is split into small, meaningful pieces." },
  { id: "embeddings", label: "Embeddings", icon: "🧬", color: "#ec4899", info: "Each chunk is converted into a vector (a list of numbers)." },
  { id: "vectordb", label: "Vector DB", icon: "🗄️", color: "#14b8a6", info: "The vectors are stored in a database for fast similarity search." },
];

export const ROUTE_LABEL = {
  none: "No search",
  single: "Single search",
  multi: "Multi-step",
  unknown: "Unknown route",
};

