const ROUTES = {
  none: { label: "No search", text: "The question was answered directly by the model. No document search was needed." },
  single: { label: "Single search", text: "The documents were searched once to build the answer." },
  multi: { label: "Multi-step", text: "The question was broken into steps, with repeated searches to build the answer." },
  unknown: { label: "Unknown route", text: "The backend did not send route information." },
};

export default function RouteBadge({ route }) {
  const key = ROUTES[route] ? route : "unknown";
  const info = ROUTES[key];
  return (
    <div className="card a-blue">
      <h3>🧭 Step 1: Route</h3>
      <span className={`badge badge-${key}`}>{info.label}</span>
      <p className="muted">{info.text}</p>
    </div>
  );
}
