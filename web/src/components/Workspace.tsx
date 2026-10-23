import { useRef, useState } from "react";

import { useTheme } from "../theme";

import type { DayDoc, DemoIndex, RouteDoc } from "../types";
import { Guide } from "./Guide";
import { HelpChat } from "./HelpChat";
import { Race } from "./Race";
import { RouteCompare } from "./RouteCompare";
import { StationDay } from "./StationDay";

export type Mode = "compare" | "race" | "day";

/** What the chat's guided example asks the demo to show. */
export type Show = (target: { mode: Mode; route?: string; day?: string; budget?: number; replay?: boolean }) => void;

const MODES: { id: Mode; title: string; sub: string }[] = [
  { id: "compare", title: "Compare two plans", sub: "One route, two plans, replayed" },
  { id: "race", title: "The race", sub: "Same computing time for all" },
  { id: "day", title: "A whole day", sub: "A station's fleet, re-planned" },
];

interface Props {
  index: DemoIndex;
  routeId: string | undefined;
  onRoute: (id: string) => void;
  route: RouteDoc | null;
  dayId: string | undefined;
  onDay: (id: string) => void;
  day: DayDoc | null;
}

/** One screen: help and chat on the left, the demo with its three modes in the middle, the guide
 * for the current mode on the right. */
export function Workspace({ index, routeId, onRoute, route, dayId, onDay, day }: Props) {
  const [mode, setMode] = useState<Mode>("compare");
  const [budget, setBudget] = useState(1);
  const [replay, setReplay] = useState<string>(); // route or day id the tour wants replayed
  const main = useRef<HTMLDivElement>(null);

  const show: Show = ({ mode: m, route: r, day: d, budget: b, replay }) => {
    setMode(m);
    if (r) onRoute(r);
    if (d) onDay(d);
    if (b) setBudget(b);
    setReplay(replay ? (m === "day" ? d : r) : undefined);
    const top = main.current?.getBoundingClientRect().top ?? 0;
    if (top < 0 || top > window.innerHeight * 0.6) main.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const dark = useTheme() === "dark";
  const shared = { routes: index.routes, routeId, onRoute, doc: route, dark };
  return (
    <section className="block" id="demo">
      <div className="sec-head">
        <h2>
          Try it <i>yourself</i>
        </h2>
        <span className="num">02</span>
      </div>
      <p className="lead">
        Real routes the model never saw during training. Nothing is computed while you browse: every plan was solved
        once and saved, so what you see is exactly what each solver produced.
      </p>
      <div className="workspace">
        <div className="ws-side ws-help">
          <HelpChat index={index} mode={mode} show={show} />
        </div>
        <div className="ws-main" ref={main}>
          <div className="modes" role="group" aria-label="Demo mode">
            {MODES.map((m) => (
              <button key={m.id} type="button" aria-pressed={mode === m.id} onClick={() => setMode(m.id)}>
                {m.title}
                <small>{m.sub}</small>
              </button>
            ))}
          </div>
          {mode === "compare" && <RouteCompare {...shared} replay={replay} onReplayed={() => setReplay(undefined)} />}
          {mode === "race" && <Race {...shared} budget={budget} onBudget={setBudget} />}
          {mode === "day" && (
            <StationDay
              days={index.days}
              dayId={dayId}
              onDay={onDay}
              doc={day}
              dark={dark}
              replay={replay}
              onReplayed={() => setReplay(undefined)}
            />
          )}
        </div>
        <div className="ws-side ws-guide">
          <Guide mode={mode} />
        </div>
      </div>
    </section>
  );
}
