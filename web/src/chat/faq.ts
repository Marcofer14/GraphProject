// Prepared answers: the chat's fallback when the AI assistant is not connected or out of free
// quota, and the source of the suggested questions. Facts match chat/knowledge.md.

export interface Faq {
  q: string;
  keys: string[]; // words (or their starts) that point to this answer, English and Spanish
  a: string;
}

export const FAQ: Faq[] = [
  {
    q: "What does this project do, in plain words?",
    keys: ["project", "plain", "proyecto", "summary", "resumen", "about", "overview"],
    a: "It is an AI that plans the order of a delivery van's stops. A neural network proposes routes and classic optimisation polishes them. On 200 real Amazon routes it had never seen, its plans need 10.5% less driving than the routes the real drivers took, about 20 minutes less per van per day.",
  },
  {
    q: "How does it compare with OR-Tools and PyVRP?",
    keys: ["pyvrp", "tools", "ortools", "compar", "better", "mejor", "solver", "benchmark", "equal", "igual"],
    a: "Given the same computing time it is ahead of both: with 1 second it is 1.19% above the best route found, against 1.60% for PyVRP and 9.17% for OR-Tools; with 2 seconds 0.77% against 1.28% and 8.09%. Given 10 seconds, PyVRP is still slightly better on average (0.70%).",
  },
  {
    q: "How was the model trained?",
    keys: ["train", "learn", "entren", "aprend", "reinforce", "refuerzo", "pomo", "curriculum", "epoch"],
    a: "With reinforcement learning, without example solutions: for each problem it builds one route from every possible first stop, and each route is pushed up or down by how much shorter or longer it is than their average (REINFORCE with the POMO baseline). It went from 50- and 100-stop synthetic problems calibrated on Amazon data to slices of real routes with real driving times.",
  },
  {
    q: "What hardware does it need?",
    keys: ["hardware", "gpu", "1660", "fast", "speed", "rapid", "veloc", "tarda", "seconds", "segundos", "cpu"],
    a: "Everything, training included, ran on a GTX 1660 Super, a mid-range gaming card from 2019. A ~150-stop route takes about half a second on it and about 2 seconds on a CPU; a whole station-day takes a few seconds.",
  },
  {
    q: "What is special about the neural network?",
    keys: ["network", "red", "neural", "transformer", "attention", "atenc", "architecture", "arquitect", "matrix", "matriz", "special", "especial"],
    a: "It is a Transformer (6 layers, 8 attention heads, 1.3 M parameters) whose attention is biased by the real one-way driving times between stops, not straight-line distance. On real streets that took the network alone from 38% to under 10% above the best route.",
  },
  {
    q: "What data was used?",
    keys: ["data", "datos", "dataset", "amazon", "routes", "rutas", "cities", "ciudades"],
    a: "The Amazon Last Mile Routing Research Challenge dataset (AWS Open Data): 9,164 real routes from 2018 in five US cities, about 1.34 million stops, with the real driving times between stops and the order each driver actually drove. Results are measured on routes never used in training.",
  },
  {
    q: "Can it plan a whole station's day?",
    keys: ["day", "dia", "station", "estac", "fleet", "flota", "vans", "furgon"],
    a: "Yes: it splits the day's stops into one group per van, routes each with the network and trades stops between vans. On 20 real days it never saw, that is 9.9% fewer fleet hours than the drivers, in a few seconds; PyVRP given 60 seconds wins only the smaller half of the days.",
  },
  {
    q: "What are its limits?",
    keys: ["limit", "weak", "debil", "problem", "contra", "cons", "downside"],
    a: "Given 10 seconds or more, PyVRP is still slightly better on average. The network alone is about 10% above the best route, so local search does much of the work. Time windows are soft and rules like driver breaks are not modelled, and drivers optimise things the data lacks (parking, access, customers). One dataset: Amazon, five US cities, 2018.",
  },
  {
    q: "Which technologies were used?",
    keys: ["tech", "tecnolog", "stack", "herramient", "librer", "framework", "languag", "lenguaj", "python", "pytorch", "react", "usaste", "used"],
    a: "Python, PyTorch with CUDA, Hydra and TensorBoard for the model; Google OR-Tools, PyVRP, Numba and NumPy for optimisation; pandas, Parquet and ijson for data; FastAPI, React, TypeScript, Vite and MapLibre for the web; uv, pytest (129 tests), Ruff, mypy and GitHub Actions for engineering.",
  },
  {
    q: "Where would it be useful?",
    keys: ["use", "uso", "useful", "sirve", "case", "caso", "aplic", "business", "empresa"],
    a: "Parcel and e-commerce delivery, re-planning during the day when an order arrives or a van breaks down, planning a whole station in seconds, and other delivery or field-service routing after retraining on that business's data, all without data-centre hardware.",
  },
  {
    q: "Who built it?",
    keys: ["who", "quien", "author", "autor", "marco", "contact", "contacto", "hire"],
    a: "Marco Fernández, a B.Sc. in Data Science student at UCA in Buenos Aires and data engineer at ExxonMobil, focused on reinforcement learning, LLM agents and data systems. Portfolio and contact: marcofer14.github.io",
  },
];

export const SUGGESTED = FAQ.slice(0, 4).map((f) => f.q);

const norm = (s: string) =>
  s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .split(/[^a-z0-9]+/)
    .filter((w) => w.length > 2);

/** The prepared answer that shares most words (or word starts) with the question, or null. */
export function faqAnswer(question: string): Faq | null {
  const words = norm(question);
  const exact = FAQ.find((f) => f.q.toLowerCase() === question.trim().toLowerCase());
  if (exact) return exact;
  let best: Faq | null = null;
  let bestScore = 0;
  for (const f of FAQ) {
    const score = f.keys.filter((k) => words.some((w) => w.startsWith(k) || k.startsWith(w))).length;
    if (score > bestScore) {
      best = f;
      bestScore = score;
    }
  }
  return best;
}
