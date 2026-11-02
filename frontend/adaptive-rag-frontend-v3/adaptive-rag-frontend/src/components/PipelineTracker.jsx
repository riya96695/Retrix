import { useState } from "react";

const BADGE = { done: "\u2713", skipped: "\u2013", error: "!" };

export default function PipelineTracker({ title, stages, statuses, details = [], note, accent = "a-blue" }) {
  const [selected, setSelected] = useState(null);
  const activeIdx = statuses.findIndex((s) => s === "active" || s === "error");
  const shownIdx = activeIdx >= 0 ? activeIdx : selected;
  const shown = shownIdx !== null ? stages[shownIdx] : null;

  return (
    <div className={`card ${accent}`}>
      <h3>{title}</h3>
      <div className="pipe-wrap">
        <ol className="pipe">
          {stages.map((s, i) => {
            const st = statuses[i] || "pending";
            const linked = st === "done" || st === "skipped";
            return (
              <li
                key={s.id}
                className={`pipe-item ${linked ? "linked" : ""}`}
                style={{ "--c": s.color, "--cn": stages[i + 1]?.color || s.color }}
              >
                <button
                  type="button"
                  className={`pipe-step st-${st} ${shownIdx === i ? "sel" : ""}`}
                  onClick={() => setSelected(i)}
                  title={s.info}
                >
                  <span className="pipe-dot">
                    {st === "active" && <span className="ring" />}
                    <span className="pipe-emoji">{s.icon}</span>
                    {BADGE[st] && <span className={`pipe-badge b-${st}`}>{BADGE[st]}</span>}
                  </span>
                  <span className="pipe-label">{s.label}</span>
                  {details[i] && <span className="pipe-detail">{details[i]}</span>}
                </button>
              </li>
            );
          })}
        </ol>
      </div>
      <div className="pipe-info" style={{ "--c": shown ? shown.color : "var(--border)" }}>
        {shown ? (
          <>
            <strong>{shown.label}:</strong> {shown.info}
          </>
        ) : (
          "Click any step to see what happens there."
        )}
      </div>
      {note && <p className="muted small">{note}</p>}
    </div>
  );
}
