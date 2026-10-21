import { useState } from "react";

const EXAMPLES = [
  "What is RAG?",
  "Compare Adaptive-RAG and CRAG",
  "How does confidence calibration help routing?",
];

export default function QueryForm({ loading, onSubmit, simulateError, onToggleError }) {
  const [text, setText] = useState("");
  const [warn, setWarn] = useState(false);

  function submit() {
    if (!text.trim()) return setWarn(true);
    setWarn(false);
    onSubmit(text.trim());
  }

  return (
    <div className="card">
      <h2 className="card-title">✨ Ask a question</h2>
      <textarea
        value={text}
        placeholder="Type your question here..."
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => (e.ctrlKey || e.metaKey) && e.key === "Enter" && submit()}
      />
      {warn && <div className="hint-err">Please enter a question first.</div>}
      <div className="chips">
        <span className="chips-label">Try:</span>
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" onClick={() => setText(ex)}>{ex}</button>
        ))}
      </div>
      <div className="row between">
        <label className="sim">
          <input type="checkbox" checked={simulateError} onChange={(e) => onToggleError(e.target.checked)} /> Simulate error
        </label>
        <button className="pri" disabled={loading} onClick={submit}>
          {loading ? "Working..." : "Submit  →"}
        </button>
      </div>
    </div>
  );
}
