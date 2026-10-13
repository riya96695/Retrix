// Circular gauge for LLM confidence
export default function ConfidenceRing({ value }) {
  const r = 44, c = 2 * Math.PI * r;
  const color = value >= 0.7 ? "#10b981" : value >= 0.45 ? "#f59e0b" : "#ef4444";
  return (
    <svg width="120" height="120" viewBox="0 0 120 120" className="ring">
      <circle cx="60" cy="60" r={r} fill="none" stroke="#e5e7f5" strokeWidth="10" />
      <circle cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
        strokeDasharray={`${c * value} ${c}`} transform="rotate(-90 60 60)" />
      <text x="60" y="67" textAnchor="middle" fontSize="22" fontWeight="800" fill="#1e1b4b">{Math.round(value * 100)}%</text>
    </svg>
  );
}
