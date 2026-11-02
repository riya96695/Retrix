const ROUTES = {
  none: {
    label: "No search",
    color: "gray",
    text: "Sawaal seedha model se answer ho gaya, document search ki zaroorat nahi padi.",
  },
  single: {
    label: "Single search",
    color: "blue",
    text: "Ek baar documents me search karke answer banaya gaya.",
  },
  multi: {
    label: "Multi-step",
    color: "purple",
    text: "Sawaal kai steps me tod ke baar-baar search karke answer banaya gaya.",
  },
  unknown: {
    label: "Unknown route",
    color: "gray",
    text: "Backend ne route ki jaankari nahi bheji.",
  },
};

export default function RouteBadge({ route }) {
  const info = ROUTES[route] || ROUTES.unknown;
  return (
    <div className="card">
      <h3>Step 1: Route</h3>
      <span className={`badge badge-${info.color}`}>{info.label}</span>
      <p className="muted">{info.text}</p>
    </div>
  );
}
