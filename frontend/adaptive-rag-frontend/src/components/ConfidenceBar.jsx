export default function ConfidenceBar({ value }) {
  const isNull = value === null || value === undefined || Number.isNaN(value);
  const pct = isNull ? 0 : Math.round(value);
  const level = pct >= 70 ? "high" : pct >= 40 ? "mid" : "low";

  return (
    <div className="card">
      <h3>Step 2: Confidence</h3>
      {isNull ? (
        <p className="muted">N/A (is route pe search nahi hua)</p>
      ) : (
        <>
          <div className="bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <div className={`bar-fill bar-${level}`} style={{ width: `${pct}%` }} />
          </div>
          <p className="pct">{pct}%</p>
        </>
      )}
    </div>
  );
}
