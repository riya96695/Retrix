export default function SourceList({ sources }) {
  if (!sources.length) return <div className="empty">No retrieval on this route, so there are no sources.</div>;
  return sources.map((s, i) => (
    <div className="src" key={s.chunk_id}>
      <div className="src-num">{i + 1}</div>
      <div className="src-body">
        <div className="meta">
          <span className="doc">📄 {s.doc} · {s.chunk_id}</span>
          <span className="score">score {s.score.toFixed(2)}</span>
        </div>
        <p>{s.text}</p>
      </div>
    </div>
  ));
}
