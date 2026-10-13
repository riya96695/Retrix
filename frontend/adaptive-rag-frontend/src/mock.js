// Mock response - shape wahi hai jo real backend se aani chahiye
export function mockAnswer(q) {
  const simple = q.length < 35;
  const complex = /\b(compare|why|how|difference|vs)\b/i.test(q) || q.length > 90;

  let probabilities, route, llm_confidence, verdict, score;
  if (simple) {
    probabilities = { no_retrieval: 0.71, single_step: 0.22, multi_step: 0.07 };
    route = "no_retrieval"; llm_confidence = 0.83; verdict = "not_used"; score = null;
  } else if (complex) {
    probabilities = { no_retrieval: 0.06, single_step: 0.24, multi_step: 0.7 };
    route = "multi_step"; llm_confidence = 0.41; verdict = "ambiguous"; score = 0.58;
  } else {
    probabilities = { no_retrieval: 0.12, single_step: 0.68, multi_step: 0.2 };
    route = "single_step"; llm_confidence = 0.52; verdict = "correct"; score = 0.87;
  }

  return {
    route,
    probabilities,
    llm_answer: "RAG connects an LLM to external documents so answers stay grounded.",
    llm_confidence,
    evaluator: { verdict, score },
    final_answer:
      route === "no_retrieval"
        ? "RAG retrieves external knowledge before answering. The model was confident enough, so retrieval was skipped."
        : "Adaptive RAG picks a route using query complexity and calibrated confidence. Retrieved chunks are graded by an evaluator; weak evidence triggers query reformulation or web fallback.",
    sources:
      route === "no_retrieval"
        ? []
        : [
            { doc: "adaptive_rag_paper.pdf", chunk_id: "c-014", score: 0.91, text: "Adaptive-RAG routes a query to no retrieval, single-step or iterative multi-step retrieval based on estimated question complexity." },
            { doc: "crag_notes.txt", chunk_id: "c-003", score: 0.84, text: "A lightweight retrieval evaluator scores retrieved documents and triggers Correct, Incorrect or Ambiguous actions." },
            { doc: "adaptive_rag_paper.pdf", chunk_id: "c-021", score: 0.66, text: "Calibration aligns stated confidence with empirical correctness, making thresholds safe for routing." },
          ],
  };
}
