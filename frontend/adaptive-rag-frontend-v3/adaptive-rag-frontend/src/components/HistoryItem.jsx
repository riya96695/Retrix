import AnswerCard from "./AnswerCard.jsx";
import RouteBadge from "./RouteBadge.jsx";
import ConfidenceBar from "./ConfidenceBar.jsx";
import ChunkList from "./ChunkList.jsx";
import { ROUTE_LABEL } from "../config.js";
import { formatMs, queryTerms } from "../utils.js";

export default function HistoryItem({ entry, open, onToggle, onTyped }) {
  const r = entry.result;
  const time = formatMs(entry.ms);
  const key = ROUTE_LABEL[r.route] ? r.route : "unknown";
  return (
    <section className="hist">
      <button type="button" className="hist-head" onClick={onToggle} aria-expanded={open}>
        <span className={`dot dot-${key}`} />
        <span className="hist-q">{entry.question}</span>
        <span className="hist-meta">
          {ROUTE_LABEL[key]}
          {time ? ` \u00b7 ${time}` : ""}
        </span>
        <span aria-hidden="true">{open ? "\u25be" : "\u25b8"}</span>
      </button>
      {open && (
        <div className="hist-body">
          <AnswerCard entry={entry} onTyped={onTyped} />
          <div className="grid2">
            <RouteBadge route={r.route} />
            <ConfidenceBar value={r.confidence} />
          </div>
          <ChunkList chunks={r.chunks} terms={queryTerms(entry.question)} />
        </div>
      )}
    </section>
  );
}
