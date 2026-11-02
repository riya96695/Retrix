export default function ExampleQuestions({ questions, onPick, disabled }) {
  return (
    <div className="examples">
      <span className="muted small">Try an example:</span>
      {questions.map((q, i) => (
        <button key={q} type="button" className={`chip c${i % 4}`} disabled={disabled} onClick={() => onPick(q)}>
          {q}
        </button>
      ))}
    </div>
  );
}
