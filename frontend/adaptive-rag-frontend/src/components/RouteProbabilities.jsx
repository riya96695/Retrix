const LABELS = { no_retrieval: "No retrieval", single_step: "Single-step", multi_step: "Multi-step" };

export default function RouteProbabilities({ probabilities }) {
  const top = Object.keys(probabilities).reduce((a, b) => (probabilities[a] > probabilities[b] ? a : b));
  return (
    <div>
      {Object.entries(probabilities).map(([key, p]) => (
        <div className="bar" key={key}>
          <span className={key === top ? "bar-label strong" : "bar-label"}>{LABELS[key]}</span>
          <div className="track">
            <div className={`fill ${key}${key === top ? " top" : ""}`} style={{ width: `${Math.round(p * 100)}%` }} />
          </div>
          <span className="bar-val">{Math.round(p * 100)}%</span>
        </div>
      ))}
    </div>
  );
}
