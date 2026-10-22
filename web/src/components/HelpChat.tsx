import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { askAssistant, CHAT_URL, MAX_CHARS, MAX_TURNS, Rich } from "../chat/assistant";
import { FAQ } from "../chat/faq";
import { loadRoute } from "../data";
import { hours, pct } from "../format";
import { bestTravel, raceSeries, resultAt } from "../race";
import type { DemoIndex, RouteDoc } from "../types";
import type { Mode, Show } from "./Workspace";

// the demo route whose saving against the driver is closest to the 200-route average
export const EXAMPLE_ROUTE = "RouteID_c9d4386d-f351-45ab-8a62-b359a77f1190";

type StepId = "howto" | "example1" | "example2" | "example3" | "looking" | "why" | "limits";

interface Msg {
  role: "user" | "assistant";
  content: string;
  note?: string;
  next?: StepId[];
}

const LABELS: Record<StepId, string> = {
  howto: "How do I use this demo?",
  example1: "Show me an example",
  example2: "Next: give every solver one second",
  example3: "Next: plan a whole day",
  looking: "What am I looking at?",
  why: "Why is the model so good?",
  limits: "What are its limits?",
};
const QUICK: StepId[] = ["howto", "example1", "looking", "why", "limits"];

const LOOKING: Record<Mode, string> = {
  compare:
    "Two maps of the **same delivery route**, each planned differently. The dots are the stops, the line is the order the van visits them, the dark dot is the station. Under each map: total driving time, the difference with the other plan, deliveries that arrived late, and how long the plan took to compute. Lines join stops in order; they are not the streets driven.",
  race:
    "A race between route planners given **the same computing time**. The chart shows, for each solver, how many extra minutes of driving its route has compared with the best route anyone found, as it gets more time (left to right). The three maps show the route each one has at the time you selected.",
  day: "A **whole day of a delivery station**: every stop it delivered, re-planned from scratch with the same vans. Each colour is one van. Under each map: total driving of the whole fleet, how many vans, late deliveries and computing time.",
};

const mins = (m: number | null) => (m === null ? "nothing yet" : m === 0 ? "the best route found" : `${m} min above the best`);

interface Props {
  index: DemoIndex;
  mode: Mode;
  show: Show;
}

export function HelpChat({ index, mode, show }: Props) {
  const [example, setExample] = useState<RouteDoc | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);
  const log = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLElement>(null);

  useEffect(() => {
    loadRoute(EXAMPLE_ROUTE).then(setExample, () => setExample(null));
  }, []);
  useEffect(() => {
    const onOpen = () => {
      box.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      setTimeout(() => input.current?.focus(), 400);
    };
    window.addEventListener("open-chat", onOpen);
    return () => window.removeEventListener("open-chat", onOpen);
  }, []);
  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  const largestDay = useMemo(() => [...index.days].sort((a, b) => b.stops - a.stops)[0], [index]);

  /** Scripted answers: free (no tokens), exact numbers from the data, and they drive the demo. */
  const step = (id: StepId): Msg => {
    const s = index.headline.solvers;
    switch (id) {
      case "howto":
        return {
          role: "assistant",
          content:
            "The demo has three modes, picked with the buttons above the maps:\n- **Compare two plans**: one real route planned two ways, with a replay of the driver's shift.\n- **The race**: every solver gets the same computing time; who has the best route after 1, 2 or 10 seconds?\n- **A whole day**: every stop of a station's day, re-planned with the same vans.\nThe guide on the right explains the controls of the mode you are in. Want me to walk you through a real example? I'll operate the demo for you.",
          next: ["example1"],
        };
      case "example1": {
        if (!example) return { role: "assistant", content: "The example is still loading; try again in a second." };
        const d = example.solvers.driver;
        const o = example.solvers.ours;
        show({ mode: "compare", route: EXAMPLE_ROUTE, replay: true });
        return {
          role: "assistant",
          content: `I've opened a typical route in **Compare** and started the replay: ${example.city}, ${example.stops.length} stops (its saving is the closest to the average of all 200 test routes).\n- Left map, the **real driver**: ${hours(d.travel_h)} of driving.\n- Right map, **our model**: ${hours(o.travel_h)}, ${pct(100 * (1 - o.travel_h / d.travel_h))} less, computed in ${o.runtime_s.toFixed(1)} s.\nSame stops, same van; only the order changes. Watch our van get back to the station first.`,
          next: ["example2"],
        };
      }
      case "example2": {
        if (!example) return { role: "assistant", content: "The example is still loading; try again in a second." };
        const best = bestTravel(example);
        const at = (key: string, budget: number) => {
          const r = resultAt(raceSeries(example, "gpu").find((x) => x.key === key)!, budget);
          return r ? Math.round((r.sol.travel_h - best) * 60) : null;
        };
        show({ mode: "race", route: EXAMPLE_ROUTE, budget: 1 });
        return {
          role: "assistant",
          content: `Now **the race** on the same route, with one second of computing each:\n- Our model: ${mins(at("ours", 1))}.\n- PyVRP: ${mins(at("pyvrp", 1))}.\n- OR-Tools: ${mins(at("ortools", 1))}.\nGiven 10 seconds, PyVRP gets to ${mins(at("pyvrp", 10))}. Over all ${index.headline.routes} test routes, at equal time our model is ahead on average; PyVRP needs about 10 seconds to catch up (${pct(s.pyvrp.gap_ctrl_pct, 2)} above the best, against ${pct(s.ours_ils2.gap_ctrl_pct, 2)} for ours with 2 s). Try the slider yourself.`,
          next: ["example3"],
        };
      }
      case "example3": {
        if (!largestDay) return { role: "assistant", content: "No station-days are loaded." };
        const h = largestDay.hours;
        show({ mode: "day", day: largestDay.id, replay: true });
        return {
          role: "assistant",
          content: `Last, **a whole day**: the largest one here, ${largestDay.city}, ${largestDay.stops.toLocaleString("en")} stops, re-planned with the same vans. I've started the replay; each colour is a van.\n- Real drivers: ${hours(h.driver)} of driving in total.\n- Our model: ${hours(h.ours)}, ${pct(100 * (1 - h.ours / h.driver))} less, in a few seconds.\n- PyVRP, given a full minute: ${hours(h.pyvrp)}${h.pyvrp > h.driver ? ", even more than the drivers: a day this size is too big for it to converge in a minute" : ""}.`,
          next: ["why"],
        };
      }
      case "looking":
        return { role: "assistant", content: LOOKING[mode], next: ["howto"] };
      case "why":
        setTimeout(() => document.getElementById("why")?.scrollIntoView({ behavior: "smooth" }), 600);
        return {
          role: "assistant",
          content: `Three things together:\n- **It reads real driving times.** Its attention is biased by the real one-way times between stops, not straight-line distance, which took the network alone from 38% to under 10% above the best route.\n- **It learned to plan, not to copy.** Reinforcement learning taught it strong starting routes, ~1,200 of them in one pass.\n- **Classic search finishes the job.** Local search polishes the best candidates and keeps improving while time is left.\nResult: ${pct(100 * (1 - s.ours.travel_h / s.driver.travel_h))} less driving than the drivers, ahead of PyVRP and OR-Tools at equal time, on a 2019 gaming GPU. I've scrolled to the details below.`,
          next: ["limits"],
        };
      case "limits":
        return { role: "assistant", content: FAQ.find((f) => f.q === "What are its limits?")!.a, next: ["howto"] };
    }
  };

  const ask = (id: StepId) => {
    if (busy) return;
    const reply = step(id); // outside the state updater: it may operate the demo
    setMessages((m) => [...m, { role: "user", content: LABELS[id] }, reply]);
  };

  const send = async (text: string) => {
    const q = text.trim().slice(0, MAX_CHARS);
    if (!q || busy) return;
    const history: Msg[] = [...messages, { role: "user", content: q }];
    setMessages(history);
    setDraft("");
    setBusy(true);
    const { answer, note } = await askAssistant(history.map(({ role, content }) => ({ role, content })));
    setMessages([...history, { role: "assistant", content: answer, note }]);
    setBusy(false);
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    void send(draft);
  };

  return (
    <section className="side-card chat" id="help" ref={box} aria-label="Help and questions">
      <div className="chat-head">
        <p className="eyebrow">Ask the project</p>
        <h3>Do you need help understanding?</h3>
        <p>
          The questions below answer instantly and can operate the demo for you.
          {CHAT_URL ? " You can also type your own; a free AI answers from the project's write-up." : ""}
        </p>
      </div>
      <div className="chat-log" ref={log} aria-live="polite">
        <div className="chat-quick">
          {QUICK.map((id) => (
            <button key={id} type="button" onClick={() => ask(id)}>
              {LABELS[id]}
            </button>
          ))}
        </div>
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            {m.role === "assistant" ? <Rich text={m.content} /> : <p>{m.content}</p>}
            {m.note && <small>{m.note}</small>}
            {m.next && i === messages.length - 1 && (
              <div className="chat-next">
                {m.next.map((id) => (
                  <button key={id} type="button" onClick={() => ask(id)}>
                    {LABELS[id]}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        {busy && <div className="chat-msg assistant typing">Thinking…</div>}
      </div>
      <form className="chat-form" onSubmit={submit}>
        <textarea
          ref={input}
          rows={2}
          value={draft}
          maxLength={MAX_CHARS}
          placeholder={CHAT_URL ? "Ask anything about the project" : "Ask a question"}
          aria-label="Your question"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send(draft);
            }
          }}
        />
        <button type="submit" className="primary" disabled={busy || !draft.trim()}>
          Ask
        </button>
      </form>
      <p className="chat-foot">
        {CHAT_URL ? "AI answers can be wrong; the numbers on this page are the reference. " : ""}
        {messages.length >= MAX_TURNS && (
          <button type="button" onClick={() => setMessages([])}>
            Start over
          </button>
        )}
      </p>
    </section>
  );
}
