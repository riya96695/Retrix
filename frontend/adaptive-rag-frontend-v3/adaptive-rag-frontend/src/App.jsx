import { useEffect, useRef, useState } from "react";
import { askQuestion } from "./api.js";
import { CancelledError } from "./errors.js";
import { EXAMPLE_QUESTIONS, QUERY_STAGES, USE_MOCK } from "./config.js";
import useTheme from "./hooks/useTheme.js";
import useStageTicker from "./hooks/useStageTicker.js";
import { buildStageMeta, formatMs, hasBackendStages, loadJSON, saveJSON, statusesFromActive } from "./utils.js";
import DocsPanel from "./components/DocsPanel.jsx";
import ExampleQuestions from "./components/ExampleQuestions.jsx";
import PipelineTracker from "./components/PipelineTracker.jsx";
import HistoryItem from "./components/HistoryItem.jsx";

const MAX_HISTORY = 30;

function Stat({ icon, label, value, color }) {
  return (
    <div className="stat" style={{ "--c": color }}>
      <div className="stat-icon">{icon}</div>
      <div>
        <div className="stat-val">{value}</div>
        <div className="stat-label">{label}</div>
      </div>
    </div>
  );
}

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const [docCount, setDocCount] = useState(0);
  const [history, setHistory] = useState(() => loadJSON("rag_history", []));
  const [openId, setOpenId] = useState(() => loadJSON("rag_history", [])[0]?.id ?? null);
  const [lastPipe, setLastPipe] = useState(null);
  const ticker = useStageTicker(5, 900); // waits at the LLM step (index 5) until the answer arrives
  const abortRef = useRef(null);
  const clockRef = useRef(null);
  const lastQuestionRef = useRef("");

  useEffect(() => {
    saveJSON("rag_history", history.slice(0, MAX_HISTORY).map((h) => ({ ...h, fresh: false })));
  }, [history]);

  useEffect(() => () => clearInterval(clockRef.current), []);

  async function ask(raw) {
    const text = String(raw || "").trim();
    if (!text || loading) return;
    lastQuestionRef.current = text;
    setLoading(true);
    setError("");
    setElapsed(0);
    setLastPipe(null);
    const controller = new AbortController();
    abortRef.current = controller;
    ticker.start();
    clockRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);

    try {
      const result = await askQuestion(text, { signal: controller.signal });
      const meta = buildStageMeta(result);
      const entry = {
        id: String(Date.now()),
        question: text,
        result,
        ms: result.timeMs ?? result.clientMs,
        ts: Date.now(),
        fresh: true,
      };
      setHistory((h) => [entry, ...h.map((x) => ({ ...x, fresh: false }))].slice(0, MAX_HISTORY));
      setOpenId(entry.id);
      setLastPipe({
        statuses: meta.map((m) => m.status),
        details: meta.map((m) => m.detail),
        estimated: !hasBackendStages(result),
      });
      setQuestion("");
    } catch (err) {
      if (!(err instanceof CancelledError)) {
        setError(err.message || "Something went wrong.");
        setLastPipe({
          statuses: statusesFromActive(QUERY_STAGES.length, ticker.getActive(), "error"),
          details: [],
          estimated: true,
        });
      }
    } finally {
      clearInterval(clockRef.current);
      ticker.reset();
      abortRef.current = null;
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask(question);
    }
  }

  function markTyped(id) {
    setHistory((h) => h.map((x) => (x.id === id ? { ...x, fresh: false } : x)));
  }

  function clearHistory() {
    if (window.confirm("Clear all history?")) {
      setHistory([]);
      setOpenId(null);
    }
  }

  const pending = QUERY_STAGES.map(() => "pending");
  const statuses = loading ? statusesFromActive(QUERY_STAGES.length, ticker.active) : lastPipe?.statuses ?? pending;
  const details = loading ? [] : lastPipe?.details ?? [];
  const showNote = loading || lastPipe?.estimated;

  const times = history.map((h) => h.ms).filter((v) => Number.isFinite(v));
  const avgMs = times.length ? times.reduce((a, b) => a + b, 0) / times.length : null;
  const confs = history.map((h) => h.result?.confidence).filter((v) => v !== null && v !== undefined && Number.isFinite(v));
  const avgConf = confs.length ? Math.round(confs.reduce((a, b) => a + b, 0) / confs.length) : null;

  return (
    <main className="container">
      <header className="hero">
        <div>
          <h1>
            Adaptive RAG <span className="hero-tag">Dashboard</span>
          </h1>
          <p>Watch how your question travels through retrieval, reranking and the LLM, step by step.</p>
        </div>
        <button type="button" className="hero-btn" onClick={toggleTheme}>
          {theme === "dark" ? "☀️ Light mode" : "🌙 Dark mode"}
        </button>
      </header>

      {USE_MOCK && (
        <div className="banner">
          🧪 Demo mode: this is sample data, not a real backend. Set VITE_USE_MOCK=false in .env to use your real backend.
        </div>
      )}

      <section className="stats">
        <Stat icon="📄" label="Documents" value={docCount} color="#3b82f6" />
        <Stat icon="💬" label="Questions asked" value={history.length} color="#8b5cf6" />
        <Stat icon="⚡" label="Avg. response time" value={formatMs(avgMs) ?? "\u2014"} color="#f59e0b" />
        <Stat icon="🎯" label="Avg. confidence" value={avgConf !== null ? `${avgConf}%` : "\u2014"} color="#22c55e" />
      </section>

      <DocsPanel onCountChange={setDocCount} />

      <section className="card a-indigo">
        <h3>💬 Ask a question</h3>
        <ExampleQuestions questions={EXAMPLE_QUESTIONS} onPick={ask} disabled={loading} />
        <form
          className="ask"
          onSubmit={(e) => {
            e.preventDefault();
            ask(question);
          }}
        >
          <textarea
            rows={3}
            placeholder="Ask a question about your documents... (Enter to send, Shift+Enter for a new line)"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <div className="row">
            <button type="submit" disabled={loading || !question.trim()}>
              {loading ? "Thinking..." : "Ask"}
            </button>
            {loading && (
              <button type="button" className="ghost" onClick={() => abortRef.current?.abort()}>
                Stop
              </button>
            )}
          </div>
        </form>
      </section>

      <PipelineTracker
        title="🔁 Query pipeline"
        accent="a-pink"
        stages={QUERY_STAGES}
        statuses={statuses}
        details={details}
        note={showNote ? "Estimated: the backend does not send stage-by-stage progress, so this animation is an estimate." : null}
      />

      {loading && (
        <div className="card loading">
          <div className="spinner" />
          <p>
            Waiting for the answer... ({elapsed}s)
            {elapsed > 15 && <span className="muted"> The model is running slowly, please wait.</span>}
          </p>
        </div>
      )}

      {error && (
        <div className="card error" role="alert">
          <strong>Error:</strong> {error}
          <div className="row" style={{ marginTop: 8 }}>
            <button type="button" className="ghost" onClick={() => ask(lastQuestionRef.current)}>
              Try again
            </button>
          </div>
        </div>
      )}

      {history.length > 0 && (
        <>
          <div className="row-between hist-title">
            <h2>🕘 Previous questions ({history.length})</h2>
            <button type="button" className="ghost danger" onClick={clearHistory}>
              Clear history
            </button>
          </div>
          {history.map((entry) => (
            <HistoryItem
              key={entry.id}
              entry={entry}
              open={openId === entry.id}
              onToggle={() => setOpenId(openId === entry.id ? null : entry.id)}
              onTyped={markTyped}
            />
          ))}
        </>
      )}
    </main>
  );
}
