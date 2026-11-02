import { useEffect, useRef, useState } from "react";
import { askQuestion } from "./api.js";
import RouteBadge from "./components/RouteBadge.jsx";
import ConfidenceBar from "./components/ConfidenceBar.jsx";
import ChunkList from "./components/ChunkList.jsx";
import UploadBox from "./components/UploadBox.jsx";

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const timerRef = useRef(null);

  useEffect(() => () => clearInterval(timerRef.current), []);

  async function handleSubmit(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || loading) return;

    setLoading(true);
    setError("");
    setResult(null);
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);

    try {
      setResult(await askQuestion(q));
    } catch (err) {
      setError(err.message || "Kuch gadbad ho gayi.");
    } finally {
      clearInterval(timerRef.current);
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>Adaptive RAG</h1>

      <UploadBox />

      <form onSubmit={handleSubmit} className="ask">
        <textarea
          rows={3}
          placeholder="Apna sawaal likho..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Soch raha hai..." : "Poocho"}
        </button>
      </form>

      {loading && (
        <div className="card loading">
          <div className="spinner" />
          <p>
            Answer aa raha hai... ({elapsed}s)
            {elapsed > 15 && <span className="muted"> Model slow chal raha hai, thoda ruko.</span>}
          </p>
        </div>
      )}

      {error && (
        <div className="card error" role="alert">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <>
          <div className="card">
            <h3>Answer</h3>
            <p className="answer">{result.answer || "Backend ne khaali answer bheja."}</p>
          </div>
          <RouteBadge route={result.route} />
          <ConfidenceBar value={result.confidence} />
          <ChunkList chunks={result.chunks} />
        </>
      )}
    </main>
  );
}
