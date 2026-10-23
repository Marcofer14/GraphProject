import type { ReactNode } from "react";

import { pct } from "../format";
import type { DemoIndex } from "../types";
import { ResultsTable } from "./Hero";
import { Summary } from "./StationDay";

interface Idea {
  hue: string;
  metric: ReactNode;
  unit: string;
  /** before → after, as % above the best route, drawn as two bars (shorter is better) */
  bars?: [number, number];
  title: string;
  short: ReactNode;
  detail: ReactNode;
}

/** Two bars, before and after, scaled to the larger one. */
function Bars({ from, to }: { from: number; to: number }) {
  const max = Math.max(from, to);
  return (
    <div className="why-bars" aria-hidden="true">
      <span style={{ width: `${(100 * from) / max}%` }} />
      <span style={{ width: `${Math.max(2, (100 * to) / max)}%` }} />
    </div>
  );
}

export function Why({ index }: { index: DemoIndex }) {
  const s = index.headline.solvers;
  const net = s.network.gap_ctrl_pct;
  const ideas: Idea[] = [
    {
      hue: "var(--route)",
      metric: <>38% → {pct(net, 1)}</>,
      unit: "above the best route, network alone",
      bars: [38, net],
      title: "It sees real streets, not straight lines",
      short: "Every attention layer reads the real one-way driving times.",
      detail: (
        <>
          Two stops can be neighbours on the map and ten minutes apart by road: only 67% of each stop's five nearest
          neighbours in a straight line are also among its five nearest by driving time. A learned bias from the real
          times took the network from 38% to under 10% above the best route.
        </>
      ),
    },
    {
      hue: "var(--truco)",
      metric: "~1,200",
      unit: "candidate routes per pass",
      title: "It learned to plan, not to copy",
      short: "Reinforcement learning: no good routes to imitate, only rewards for shorter ones.",
      detail: (
        <>
          It plans a route from every possible first stop and rewards the ones shorter than their siblings, over
          millions of routes. At inference it repeats the trick on 8 rotated or mirrored copies of the map: about 1,200
          candidates in a single GPU pass.
        </>
      ),
    },
    {
      hue: "var(--uni)",
      metric: (
        <>
          {pct(net, 1)} → {pct(s.ours_ils2.gap_ctrl_pct, 2)}
        </>
      ),
      unit: "above the best route, with search",
      bars: [net, s.ours_ils2.gap_ctrl_pct],
      title: "A neural start, a classic finish",
      short: "The network is fast but rough; local search is exact but needs a good start.",
      detail: (
        <>
          Polishing the 8 best candidates gives {pct(s.ours.gap_ctrl_pct, 2)} in half a second, and searching on gives{" "}
          {pct(s.ours_ils2.gap_ctrl_pct, 2)} in 2 seconds. With the same time, PyVRP is at{" "}
          {pct(s.pyvrp_2s.gap_ctrl_pct, 2)} and OR-Tools at {pct(s.ortools_2s.gap_ctrl_pct, 2)}.
        </>
      ),
    },
    {
      hue: "var(--iberia)",
      metric: "0.5 s",
      unit: "per 150-stop route, 2019 GPU",
      title: "Engineered to be cheap",
      short: "Measured first, then optimised: 2.5× faster with identical answers.",
      detail: (
        <>
          Profiling showed the bottleneck was scoring candidates in Python, not the network; vectorising it and
          compiling the local search with Numba did the rest. About two seconds on a CPU, with no solver licences.
        </>
      ),
    },
  ];

  return (
    <section className="block" id="why">
      <div className="sec-head">
        <h2>
          Why it works <i>so well</i>
        </h2>
        <span className="num">03</span>
      </div>
      <p className="lead">
        Four ideas, each measured. Together they turn a model that fits on a gaming card into a planner that beats
        established route optimisers when all get the same time.
      </p>
      <div className="why-grid">
        {ideas.map((d, i) => (
          <article key={d.title} className="why" style={{ ["--hue" as string]: d.hue }}>
            <div className="why-top">
              <span className="why-n">{String(i + 1).padStart(2, "0")}</span>
              <div>
                <p className="why-metric">{d.metric}</p>
                <p className="why-unit">{d.unit}</p>
              </div>
            </div>
            {d.bars && <Bars from={d.bars[0]} to={d.bars[1]} />}
            <h3>{d.title}</h3>
            <p className="why-short">{d.short}</p>
            <p className="why-detail">{d.detail}</p>
          </article>
        ))}
      </div>

      <div className="subhead">
        <p className="eyebrow">Beyond one van</p>
        <h3>On whole station-days</h3>
      </div>
      <Summary days={index.days} />

      <div className="subhead">
        <p className="eyebrow">Every number</p>
        <h3>All results on {index.headline.routes} unseen routes</h3>
      </div>
      <ResultsTable index={index} />
    </section>
  );
}
