import { useState } from "react";
import { splitHighlight } from "../utils.js";

const LIMIT = 180;

function Chunk({ c, terms }) {
  const [open, setOpen] = useState(false);
  const long = c.text.length > LIMIT;
  const text = !long || open ? c.text : c.text.slice(0, LIMIT).trimEnd() + "\u2026";
  return (
    <li>
      <div className="chunk-meta">
        {c.source && <span className="tag t-blue">{c.source}</span>}
        {c.page !== null && c.page !== undefined && <span className="tag t-purple">page {c.page}</span>}
        {c.score !== null && c.score !== undefined && <span className="tag t-orange">score {c.score.toFixed(2)}</span>}
      </div>
      <p>
        {splitHighlight(text, terms).map((p, k) =>
          p.hit ? <mark key={k}>{p.t}</mark> : <span key={k}>{p.t}</span>
        )}
      </p>
      {long && (
        <button type="button" className="link" onClick={() => setOpen((o) => !o)}>
          {open ? "Show less" : "Show more"}
        </button>
      )}
    </li>
  );
}

export default function ChunkList({ chunks, terms = [] }) {
  const list = Array.isArray(chunks) ? chunks : [];
  return (
    <div className="card a-orange">
      <h3>📚 Retrieved chunks ({list.length})</h3>
      {list.length === 0 ? (
        <p className="muted" style={{ marginTop: 0 }}>No chunks were retrieved.</p>
      ) : (
        <ol className="chunks">
          {list.map((c, i) => (
            <Chunk key={`${c.id}-${i}`} c={c} terms={terms} />
          ))}
        </ol>
      )}
    </div>
  );
}
