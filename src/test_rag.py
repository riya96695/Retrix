"""End-to-end test of RagSystem.ask() with a small made-up document. Run on a GPU (Kaggle).
Natural routing is only printed. Forced-route checks test the plumbing (answer_confidence, searches, chunks).
"""
from rag import RagSystem
from route import Retrix

DOC = """The Varnor Bridge spans the Elka River in the northern town of Halvrik. Construction began in 1969 and the bridge was completed in 1974. It was designed by the engineer Ilsa Marrow, who also supervised the steel works. The bridge is 412 metres long and carries two lanes of traffic and a footpath. Local officials described it as the most ambitious public project in the region during the nineteen seventies, and it remains the main crossing between the old town and the eastern districts.

The Tessara Observatory stands on the western bank of the Elka River, a short walk from Halvrik's old town. It opened in 1988 and houses a refracting telescope donated by the Halvrik Science Society. The observatory is open to visitors on weekends and runs evening talks about astronomy for schools in the area. Its first director was the astronomer Pavel Orsk, who led a small team of four researchers during the early years.

Halvrik has a population of about thirty thousand people. The town is known for its wool market, which takes place every Thursday in the central square, and for the annual lantern festival held in November. The Halvrik Science Society was founded in 1952 by a group of teachers who wanted to bring science lectures to the town. Today it has around two hundred members and publishes a small quarterly newsletter.

The Elka River rises in the Dornak hills and flows south for ninety kilometres before reaching the sea at the port of Kessin. The river floods in spring, and a system of embankments was built in 1931 to protect Halvrik. Fishing is permitted along most of the river, although a permit from the town council is required. Several smaller bridges cross the river upstream, but the Varnor Bridge is the only one that can carry heavy lorries.

Ilsa Marrow was born in Kessin in 1931 and trained as a civil engineer. Before the Varnor Bridge she worked on railway viaducts in the Dornak hills. After the bridge opened she taught engineering at the Kessin Technical Institute until she retired in 1996. She received the Silver Compass award for public works in 1975, the same year the bridge won a regional prize for design."""

Q_SINGLE = "In what year was the Varnor Bridge completed?"
Q_MULTI = "Who designed the bridge that crosses the river next to the Tessara Observatory?"
Q_NO = "What is the capital of France?"

results = []


def check(name, ok):
    results.append((name, ok))
    print(f"CHECK {'PASS' if ok else 'FAIL'}: {name}")


def show(tag, r):
    print(f"== {tag}\n   route={r['route']} | searches={r['n_searches']} | chunks={len(r['retrieved'])}"
          f" | answer_confidence={r['answer_confidence']} | note={r['note']!r}\n   answer: {r['final_answer']}")


def force(system, route):
    orig = system.retrix.route
    system.retrix.route = lambda q: {**orig(q), "route": route}
    return orig


system = RagSystem(Retrix())
check("live confidence loaded (rag.conf is not None)", system.conf is not None)

# 1. no documents loaded
r = system.ask(Q_SINGLE); show("no documents", r)
check("no docs: note set, no chunks, answer_confidence None",
      bool(r["note"]) and not r["retrieved"] and r["answer_confidence"] is None)

n = system.store.add_text("halvrik.txt", DOC)
print(f"\nloaded halvrik.txt: {n} chunks")
check("document gives >= 2 chunks", n >= 2)

# 2. natural routing (printed only)
for q in (Q_NO, Q_SINGLE, Q_MULTI):
    show(f"natural: {q}", system.ask(q))

# 3. forced routes: plumbing checks
orig = force(system, "SINGLE_STEP")
r = system.ask(Q_SINGLE); show("forced SINGLE_STEP", r)
check("SINGLE: answer_confidence is a float in [0,1]",
      isinstance(r["answer_confidence"], float) and 0 <= r["answer_confidence"] <= 1)
check("SINGLE: 1 search, chunks returned", r["n_searches"] == 1 and len(r["retrieved"]) >= 2)
check("SINGLE: answer contains 1974", "1974" in r["final_answer"])

system.retrix.route = orig
orig = force(system, "MULTI_STEP")
r = system.ask(Q_MULTI); show("forced MULTI_STEP", r)
check("MULTI: answer_confidence None, >= 1 search", r["answer_confidence"] is None and r["n_searches"] >= 1)
check("MULTI: answer mentions Marrow", "marrow" in r["final_answer"].lower())

system.retrix.route = orig
orig = force(system, "NO_RETRIEVAL")
r = system.ask(Q_NO); show("forced NO_RETRIEVAL", r)
check("NO_RETRIEVAL: 0 searches, answer_confidence None", r["n_searches"] == 0 and r["answer_confidence"] is None)
system.retrix.route = orig

print(f"\n{sum(ok for _, ok in results)}/{len(results)} checks passed")