import { pct } from "../format";
import type { DemoIndex } from "../types";

const REPO = "https://github.com/Marcofer14/GraphProject";
const DECISIONS = `${REPO}/blob/main/docs/decisions.md`;

export function Limits({ index }: { index: DemoIndex }) {
  const s = index.headline.solvers;
  return (
    <section className="block" id="limits">
      <div className="sec-head">
        <h2>
          Limits, <i>honestly</i>
        </h2>
        <span className="num">06</span>
      </div>
      <ul className="limits">
        <li>
          Given 10 seconds, PyVRP is still slightly better on average ({pct(s.pyvrp.gap_ctrl_pct, 2)} above the best
          route, against {pct(s.ours_ils2.gap_ctrl_pct, 2)} for our model with 2 s). Our edge is speed: at equal time
          it is ahead.
        </li>
        <li>
          The network on its own is {pct(s.network.gap_ctrl_pct, 1)} above the best route; local search does much of
          the final work. The network's job is a good, fast starting point.
        </li>
        <li>The network runs on a GPU. On a CPU the first answer takes about 2 s instead of half a second.</li>
        <li>
          Real drivers optimise things this model does not see: parking, building access, customer habits, safety.
          Shorter driving time is not automatically a better day for the driver, and our routes look quite different
          from theirs.
        </li>
        <li>
          One dataset: Amazon deliveries in five US cities in 2018. Lines on the maps join stops in order; they are not
          street geometry.
        </li>
      </ul>
      <p className="note">
        Every decision, with its numbers, is written up in the <a href={DECISIONS}>design decisions</a> of the{" "}
        <a href={REPO}>repository</a>.
      </p>
    </section>
  );
}
