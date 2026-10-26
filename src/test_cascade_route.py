"""Sanity check of the cascade router with the live LLM. Run on a GPU (Kaggle)."""
from route import Retrix

r = Retrix()
print("cascade active:", r.cascade is not None)
for q in ["What is the capital of France?",
          "When did Beyonce start becoming popular?",
          "Were Scott Derrickson and Ed Wood of the same nationality?",
          "In what year was the Varnor Bridge completed?",
          "Who designed the bridge that crosses the river next to the Tessara Observatory?"]:
    out = r.route(q)
    print(f"{out['route']:13s} {out['probs']} | {q} -> {out['hypothetical_answer']}")