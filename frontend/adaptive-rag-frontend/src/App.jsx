import { useState } from "react";
import { askQuery } from "./api.js";
import QueryForm from "./components/QueryForm.jsx";
import ResultCard from "./components/ResultCard.jsx";
import UploadPanel from "./components/UploadPanel.jsx";

const NAV = [
  { id: "ask", label: "Ask", icon: "💬" },
  { id: "docs", label: "Documents", icon: "📄" },
];
const ROUTE_LABEL = { no_retrieval: "No retrieval", single_step: "Single-step", multi_step: "Multi-step" };

export default function App() {
  const [tab, setTab] = useState("ask");
  const [status, setStatus] = useState("idle"); // idle | loading | error | done
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [lastQuery, setLastQuery] = useState("");
  const [simulateError, setSimulateError] = useState(false);
  const [count, setCount] = useState(0);

  async function handleSubmit(query) {
    setLastQuery(query);
    setStatus("loading");
    setError("");
    try {
      setResult(await askQuery(query, { simulateError }));
      setCount((n) => n + 1);
      setStatus("done");
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-mark">R</div>
          <div><div className="logo-name">Retrix</div><div className="logo-sub">Adaptive RAG</div></div>
        </div>
        <nav>
          {NAV.map((n) => (
            <button key={n.id} className={tab === n.id ? "nav active" : "nav"} onClick={() => setTab(n.id)}>
              <span>{n.icon}</span> {n.label}
            </button>
          ))}
        </nav>
        <div className="side-foot">● Mock mode<br /><small>Backend not connected</small></div>
      </aside>

      <main className="main">
        <section className="hero">
          <h1>Adaptive RAG Dashboard</h1>
          <p>Confidence-calibrated routing with self-correcting retrieval</p>
        </section>

        <section className="stats">
          <div className="stat"><div className="stat-ico a">🧠</div><div><b>{count}</b><span>Queries run</span></div></div>
          <div className="stat"><div className="stat-ico b">🧭</div><div><b>{result ? ROUTE_LABEL[result.route] : "—"}</b><span>Last route</span></div></div>
          <div className="stat"><div className="stat-ico c">🎯</div><div><b>{result ? `${Math.round(result.llm_confidence * 100)}%` : "—"}</b><span>Last LLM confidence</span></div></div>
        </section>

        {tab === "ask" ? (
          <>
            <QueryForm loading={status === "loading"} onSubmit={handleSubmit}
              simulateError={simulateError} onToggleError={setSimulateError} />
            {status === "loading" && (
              <div className="card loading"><span className="spin" /> Routing and retrieval in progress...</div>
            )}
            {status === "error" && (
              <div className="err"><span>⚠️ {error}</span><button onClick={() => handleSubmit(lastQuery)}>Retry</button></div>
            )}
            {status === "done" && result && <ResultCard data={result} />}
          </>
        ) : (
          <UploadPanel />
        )}
      </main>
    </div>
  );
}
