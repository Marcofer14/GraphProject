// Cloudflare Worker behind the demo's "Ask the project" chat. Keeps the Groq API key off the
// page, picks the relevant parts of the knowledge base, and answers with Groq's free tier, falling
// back to Cloudflare Workers AI (free daily allowance) when Groq is unavailable or out of quota.

import knowledge from "../knowledge.md";
import { chunks, prompt, RateLimiter, validate } from "./core";

interface Ai {
  run(model: string, input: unknown): Promise<unknown>;
}

interface Env {
  GROQ_API_KEY?: string; // secret: npx wrangler secret put GROQ_API_KEY
  GROQ_MODELS: string; // comma-separated, tried in order
  WORKERS_AI_MODEL: string;
  ALLOWED_ORIGINS: string; // comma-separated
  AI?: Ai;
}

const ALL = chunks(knowledge);
const limiter = new RateLimiter(20, 10 * 60 * 1000); // 20 questions per visitor per 10 minutes
const MAX_TOKENS = 350;

type Turn = { role: string; content: string };

async function askGroq(env: Env, messages: Turn[]): Promise<string | null> {
  if (!env.GROQ_API_KEY) return null;
  for (const model of env.GROQ_MODELS.split(",").map((m) => m.trim()).filter(Boolean)) {
    const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${env.GROQ_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages, max_tokens: MAX_TOKENS, temperature: 0.2 }),
    });
    if (res.ok) {
      const data = (await res.json()) as { choices?: { message?: { content?: string } }[] };
      const answer = data.choices?.[0]?.message?.content?.trim();
      if (answer) return answer;
    } else {
      console.warn(`groq ${model}: ${res.status}`); // 429 = quota, try the next model
    }
  }
  return null;
}

async function askWorkersAi(env: Env, messages: Turn[]): Promise<string | null> {
  if (!env.AI) return null;
  try {
    const out = (await env.AI.run(env.WORKERS_AI_MODEL, { messages, max_tokens: MAX_TOKENS })) as { response?: string };
    return out.response?.trim() || null;
  } catch (err) {
    console.warn(`workers ai: ${err}`);
    return null;
  }
}

function cors(origin: string | null, env: Env): Record<string, string> {
  const allowed = env.ALLOWED_ORIGINS.split(",").map((o) => o.trim());
  return {
    "Access-Control-Allow-Origin": origin && allowed.includes(origin) ? origin : allowed[0],
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    Vary: "Origin",
  };
}

function json(body: unknown, status: number, headers: Record<string, string>): Response {
  return new Response(JSON.stringify(body), { status, headers: { ...headers, "Content-Type": "application/json" } });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const origin = request.headers.get("Origin");
    const headers = cors(origin, env);
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers });
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health") return json({ ok: true }, 200, headers);
    if (request.method !== "POST" || url.pathname !== "/chat") return json({ error: "Not found." }, 404, headers);
    if (!origin || !env.ALLOWED_ORIGINS.split(",").map((o) => o.trim()).includes(origin)) {
      return json({ error: "Origin not allowed." }, 403, headers);
    }
    const visitor = request.headers.get("CF-Connecting-IP") ?? "unknown";
    if (!limiter.allow(visitor)) {
      return json({ error: "Too many questions for now; try again in a few minutes." }, 429, headers);
    }

    let body: unknown;
    try {
      body = await request.json();
    } catch {
      return json({ error: "Send JSON." }, 400, headers);
    }
    const checked = validate(body);
    if ("error" in checked) return json({ error: checked.error }, 400, headers);

    const messages = prompt(ALL, checked.messages);
    const groq = await askGroq(env, messages);
    if (groq) return json({ answer: groq, source: "groq" }, 200, headers);
    const workers = await askWorkersAi(env, messages);
    if (workers) return json({ answer: workers, source: "workers-ai" }, 200, headers);
    return json({ error: "The assistant is out of free quota right now." }, 503, headers);
  },
};
