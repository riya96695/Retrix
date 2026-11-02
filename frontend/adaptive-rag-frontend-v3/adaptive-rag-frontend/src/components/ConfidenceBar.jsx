export default function ConfidenceBar({ value }) {
  const isNull = value === null || value === undefined || Number.isNaN(value);
  const pct = isNull ? 0 : Math.round(value);
  const level = pct >= 70 ? "high" : pct >= 40 ? "mid" : "low";
  const levelText = { high: "High", mid: "Medium", low: "Low" }[level];

  return (
    <div className="card a-green">
      <h3>🎯 Step 2: Confidence</h3>
      {isNull ? (
        <p className="muted" style={{ marginTop: 0 }}>N/A (no search was performed on this route)</p>
      ) : (
        <>
          <div className="conf-top">
            <span className="pct">{pct}%</span>
            <span className={`level lv-${level}`}>{levelText}</span>
          </div>
          <div className="bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <div className={`bar-fill bar-${level}`} style={{ width: `${pct}%` }} />
          </div>
        </>
      )}
    </div>
  );
}
