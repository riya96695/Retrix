import { useEffect, useState } from "react";
import { collectSources, formatMs } from "../utils.js";

// The answer is revealed gradually (typewriter). This is a browser effect, not backend streaming.
function useTypewriter(text, enabled) {
  const [n, setN] = useState(enabled ? 0 : text.length);
  useEffect(() => {
    if (!enabled) {
      setN(text.length);
      return undefined;
    }
    setN(0);
    const id = setInterval(() => {
      setN((v) => {
        if (v >= text.length) {
          clearInterval(id);
          return v;
        }
        return v + 3;
      });
    }, 20);
    return () => clearInterval(id);
  }, [text, enabled]);
  return { shown: text.slice(0, n), done: n >= text.length, finish: () => setN(text.length) };
}

export default function AnswerCard({ entry, onTyped }) {
  const { result, ms, fresh } = entry;
  const text = result.answer || "The backend returned an empty answer.";
  const { shown, done, finish } = useTypewriter(text, fresh);
  const [copy, setCopy] = useState("idle");
  const sources = collectSources(result.chunks);

  useEffect(() => {
    if (fresh && done && onTyped) onTyped(entry.id);
  }, [fresh, done]); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopy("ok");
    } catch {
      setCopy("fail");
    }
    setTimeout(() => setCopy("idle"), 1800);
  }

  const time = formatMs(ms);

  return (
    <div className="card a-pink answer-card">
      <div className="row-between">
        <h3>💡 Answer</h3>
        <div className="row">
          {!done && (
            <button type="button" className="ghost" onClick={finish}>
              Show full answer
            </button>
          )}
          <button type="button" className="ghost" onClick={handleCopy}>
            {copy === "ok" ? "Copied!" : copy === "fail" ? "Copy failed" : "Copy"}
          </button>
        </div>
      </div>
      <p className="answer">
        {shown}
        {!done && <span className="cursor">|</span>}
      </p>
      {done && (
        <div className="meta">
          {time && <span className="pill p-orange">⏱ Answered in {time}</span>}
          {sources.length > 0 ? (
            sources.map((s) => (
              <span key={s} className="pill p-blue">📄 {s}</span>
            ))
          ) : (
            <span className="pill p-gray">
              {result.route === "none" ? "No sources (no document search)" : "No sources found"}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
