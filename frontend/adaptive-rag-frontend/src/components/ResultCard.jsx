import RouteProbabilities from "./RouteProbabilities.jsx";
import SourceList from "./SourceList.jsx";
import ConfidenceRing from "./ConfidenceRing.jsx";

const ROUTES = { no_retrieval: "No retrieval", single_step: "Single-step", multi_step: "Multi-step" };
const ICONS = { no_retrieval: "⚡", single_step: "🔍", multi_step: "🧭" };
const VERDICTS = {
  correct: ["✔ Correct", "ok"],
  ambiguous: ["◐ Ambiguous", "warn"],
  incorrect: ["✖ Incorrect", "bad"],
  not_used: ["Evaluator skipped", "neutral"],
};

export default function ResultCard({ data }) {
  const [vLabel, vClass] = VERDICTS[data.evaluator.verdict] ?? ["-", "neutral"];
  return (
    <div className="card result">
      <div className="result-head">
        <span className={`route-pill ${data.route}`}>{ICONS[data.route]} {ROUTES[data.route]}</span>
        <span className={`badge ${vClass}`}>
          {vLabel}{data.evaluator.score != null && ` · ${data.evaluator.score.toFixed(2)}`}
        </span>
      </div>

      <div className="grid2">
        <div>
          <h3>Route probabilities</h3>
          <RouteProbabilities probabilities={data.probabilities} />
        </div>
        <div className="ring-wrap">
          <ConfidenceRing value={data.llm_confidence} />
          <div className="ring-label">LLM confidence</div>
        </div>
      </div>

      <div className="sec"><h3>LLM direct answer</h3><div className="box">{data.llm_answer}</div></div>
      <div className="sec"><h3>Final answer</h3><div className="box final">{data.final_answer}</div></div>
      <div className="sec"><h3>Sources · {data.sources.length} chunks</h3><SourceList sources={data.sources} /></div>
    </div>
  );
}
