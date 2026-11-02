export default function ChunkList({ chunks }) {
  const list = Array.isArray(chunks) ? chunks : [];
  return (
    <div className="card">
      <h3>Retrieved chunks ({list.length})</h3>
      {list.length === 0 ? (
        <p className="muted">Koi chunk retrieve nahi hua.</p>
      ) : (
        <ol className="chunks">
          {list.map((c, i) => (
            <li key={`${c.id}-${i}`}>
              <div className="chunk-meta">
                {c.source && <span className="tag">{c.source}</span>}
                {c.score !== null && <span className="tag">score {c.score.toFixed(2)}</span>}
              </div>
              <p>{c.text}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
