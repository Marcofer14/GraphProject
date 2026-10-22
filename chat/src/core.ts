// Pure logic of the project assistant: pick the relevant parts of the knowledge base, validate the
// conversation and build the prompt. No Cloudflare APIs here, so it is unit-tested with vitest.

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface Chunk {
  title: string;
  text: string;
}

export const LIMITS = { messages: 12, chars: 600, history: 6, chunks: 4 };

/** Split the knowledge base into its "## " sections. */
export function chunks(knowledge: string): Chunk[] {
  return knowledge
    .split(/\n## /)
    .slice(1)
    .map((part) => {
      const [title, ...rest] = part.split("\n");
      return { title: title.trim(), text: rest.join("\n").trim() };
    });
}

const STOP = new Set(
  "a an and are as at be but by can do does for from has have how i if in is it its me my not of on or so that the this to was what when where which who why will with you your about".split(
    " ",
  ),
);

export function words(text: string): string[] {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .split(/[^a-z0-9.%]+/)
    .map((w) => w.replace(/^\.+|\.+$/g, ""))
    .filter((w) => w.length > 1 && !STOP.has(w));
}

// Spanish (and a few loose) terms mapped to the knowledge base's English words, since visitors
// may ask in Spanish while retrieval matches words.
const SYNONYMS: Record<string, string> = {
  entrenamiento: "training", entreno: "training", entrenaron: "training", entrenado: "trained",
  aprendio: "learned", aprendizaje: "learning", refuerzo: "reinforcement", resultados: "results",
  resultado: "results", limites: "limits", limitaciones: "limits", debilidades: "limits",
  tecnologias: "stack", herramientas: "stack", lenguajes: "stack", stack: "stack", datos: "data",
  dataset: "data", modelo: "network", red: "network", neuronal: "neural", arquitectura: "network",
  conductores: "drivers", conductor: "drivers", choferes: "drivers", ruta: "route", rutas: "routes",
  velocidad: "speed", rapido: "speed", tarda: "speed", tiempo: "time", placa: "hardware",
  grafica: "gpu", autor: "author", quien: "author", contacto: "contact", casos: "use", uso: "use",
  sirve: "use", aplicaciones: "use", funciona: "inference", comparacion: "comparison",
  compara: "comparison", dia: "station-days", dias: "station-days", estacion: "station-days",
  furgonetas: "vans", camionetas: "vans", demo: "demo", pagina: "demo", reproducir: "reproducing",
  instalar: "reproducing", atencion: "attention", matriz: "matrix", problema: "problem",
  ventanas: "windows", tarde: "late", optimizacion: "optimisation", busqueda: "search",
  eficiente: "speed", similares: "similarity", parecido: "similarity",
  fast: "speed", quick: "speed", slow: "speed", runtime: "speed", efficient: "speed",
  tools: "stack", libraries: "stack", frameworks: "stack", languages: "stack",
};

// sent when a question matches nothing, so the model still has the essentials
const DEFAULTS = ["Headline results", "The neural network", "How it learned (reinforcement learning)", "Limits"];

/** Same word up to its ending: shared 5-letter start ("trained", "training"), or one word of 4+
 * letters starting the other ("tech", "technologies"). */
function related(a: string, b: string): boolean {
  if (a === b) return true;
  if (a.length >= 5 && b.length >= 5) return a.slice(0, 5) === b.slice(0, 5);
  return Math.min(a.length, b.length) >= 4 && (a.startsWith(b) || b.startsWith(a));
}

/** The overview plus the `k` sections sharing most words with the question (title words count
 * double; related forms match, see `related`). */
export function retrieve(all: Chunk[], question: string, k = LIMITS.chunks): Chunk[] {
  const q = [...new Set(words(question).flatMap((w) => (SYNONYMS[w] ? [w, ...words(SYNONYMS[w])] : [w])))];
  const score = (c: Chunk) => {
    const hits = (text: string) => words(text).filter((w) => q.some((x) => related(x, w))).length;
    return 2 * hits(c.title) + hits(c.text) / Math.sqrt(1 + words(c.text).length / 60);
  };
  const [overview, ...rest] = all;
  const ranked = rest
    .map((c) => ({ c, s: score(c) }))
    .filter((x) => x.s > 0)
    .sort((a, b) => b.s - a.s)
    .slice(0, k)
    .map((x) => x.c);
  if (ranked.length === 0) ranked.push(...rest.filter((c) => DEFAULTS.includes(c.title)));
  return [overview, ...ranked];
}

/** Checks the request body; returns the cleaned conversation or an error message. */
export function validate(body: unknown): { messages: Message[] } | { error: string } {
  const msgs = (body as { messages?: unknown })?.messages;
  if (!Array.isArray(msgs) || msgs.length === 0) return { error: "Send at least one message." };
  if (msgs.length > LIMITS.messages) return { error: "This conversation is too long; start a new one." };
  const clean: Message[] = [];
  for (const m of msgs) {
    const role = (m as Message)?.role;
    const content = (m as Message)?.content;
    if ((role !== "user" && role !== "assistant") || typeof content !== "string") {
      return { error: "Malformed message." };
    }
    if (content.length > LIMITS.chars) return { error: `Keep questions under ${LIMITS.chars} characters.` };
    clean.push({ role, content: content.trim() });
  }
  if (clean.at(-1)!.role !== "user" || !clean.at(-1)!.content) return { error: "The last message must be a question." };
  return { messages: clean };
}

export const SYSTEM = `You are the assistant on the demo page of "Neural last-mile routing", a portfolio project by Marco Fernández.
Answer questions about the project, how it works, its results and limits, the technologies used, and its author.
Rules:
- Use only the facts in the CONTEXT below. If the answer is not there, say you don't know and suggest the GitHub repository or contacting Marco. Never invent numbers, names or features.
- Be accurate about comparisons: at equal computing time the model beats PyVRP and OR-Tools on average, but PyVRP given 10 seconds is still slightly better on average.
- Explain plainly first (the reader may not be technical), then add technical detail if it helps. Keep answers under about 150 words.
- Answer in the language the user writes in.
- Politely decline unrelated requests (other topics, writing code for the user, role-play).`;

/** System prompt with the retrieved context, followed by the recent conversation. */
export function prompt(all: Chunk[], conversation: Message[]): { role: string; content: string }[] {
  const recent = conversation.slice(-LIMITS.history);
  const question = recent
    .filter((m) => m.role === "user")
    .slice(-2)
    .map((m) => m.content)
    .join(" ");
  const context = retrieve(all, question)
    .map((c) => `## ${c.title}\n${c.text}`)
    .join("\n\n");
  return [{ role: "system", content: `${SYSTEM}\n\nCONTEXT:\n${context}` }, ...recent];
}

/** Best-effort per-visitor limit inside one Worker instance (instances are short-lived, and the
 * model provider's own free-tier quota is the hard cap). */
export class RateLimiter {
  private hits = new Map<string, number[]>();
  constructor(
    private max: number,
    private windowMs: number,
  ) {}

  allow(key: string, now = Date.now()): boolean {
    const recent = (this.hits.get(key) ?? []).filter((t) => now - t < this.windowMs);
    if (recent.length >= this.max) {
      this.hits.set(key, recent);
      return false;
    }
    recent.push(now);
    this.hits.set(key, recent);
    if (this.hits.size > 5000) this.hits.clear(); // keep memory bounded
    return true;
  }
}
