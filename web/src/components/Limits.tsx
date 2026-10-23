import type { ReactNode } from "react";

import { pct } from "../format";
import type { DemoIndex } from "../types";

const REPO = "https://github.com/Marcofer14/GraphProject";
const DECISIONS = `${REPO}/blob/main/docs/decisions.md`;

export function Limits({ index }: { index: DemoIndex }) {
  const s = index.headline.solvers;
  const items: { tag: string; title: string; text: ReactNode }[] = [
    {
      tag: "Quality",
      title: "With time to spare, PyVRP still wins",
      text: (
        <>
          Given 10 seconds, PyVRP is {pct(s.pyvrp.gap_ctrl_pct, 2)} above the best route, against{" "}
          {pct(s.ours_ils2.gap_ctrl_pct, 2)} for our model with 2 s. Our edge is speed: at equal time it is ahead.
        </>
      ),
    },
    {
      tag: "Network",
      title: "Search does much of the final work",
      text: (
        <>
          The network on its own is {pct(s.network.gap_ctrl_pct, 1)} above the best route. Its job is a good, fast
          starting point for local search.
        </>
      ),
    },
    {
      tag: "Hardware",
      title: "Fastest on a GPU",
      text: "On a CPU the first answer takes about 2 s instead of half a second.",
    },
    {
      tag: "Drivers",
      title: "Shorter is not always a better day",
      text: "Drivers optimise what the model does not see: parking, building access, customer habits, safety. Our routes look quite different from theirs.",
    },
    {
      tag: "Data",
      title: "One dataset, one era",
      text: "Amazon deliveries in five US cities in 2018. Lines on the maps join stops in order; they are not street geometry.",
    },
  ];
  return (
    <section className="block" id="limits">
      <div className="sec-head">
        <h2>
          Limits, <i>honestly</i>
        </h2>
        <span className="num">06</span>
      </div>
      <p className="lead">What the results do not show, so they are read for what they are.</p>
      <ol className="limit-list">
        {items.map((l, i) => (
          <li key={l.title}>
            <span className="limit-n">{String(i + 1).padStart(2, "0")}</span>
            <div>
              <span className="limit-tag">{l.tag}</span>
              <b>{l.title}</b>
              <p>{l.text}</p>
            </div>
          </li>
        ))}
      </ol>
      <div className="callout">
        <p>
          Every decision, with its numbers, is written up in 21 design decisions: what was tried, what was measured and
          why it was kept or dropped.
        </p>
        <div className="callout-actions">
          <a className="btn primary" href={DECISIONS}>
            Read the design decisions ↗
          </a>
          <a className="btn" href={REPO}>
            Source on GitHub ↗
          </a>
        </div>
      </div>
    </section>
  );
}
