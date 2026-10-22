import { pct } from "../format";
import type { DemoIndex } from "../types";
import { ResultsTable } from "./Hero";
import { Summary } from "./StationDay";

export function Why({ index }: { index: DemoIndex }) {
  const s = index.headline.solvers;
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
        <article className="why">
          <p className="big">38% → {pct(s.network.gap_ctrl_pct, 1)}</p>
          <h3>It sees real streets, not straight lines</h3>
          <p>
            Two stops can be neighbours on the map and ten minutes apart by road. Only 67% of each stop's five nearest
            neighbours in a straight line are also among its five nearest by real driving time.
          </p>
          <p>
            So the network reads the real one-way driving times: every attention layer gets a learned bias from them.
            That alone took the network from 38% to under 10% above the best route.
          </p>
        </article>
        <article className="why">
          <p className="big">~1,200 routes</p>
          <h3>It learned to plan, not to copy</h3>
          <p>
            Nobody gave it good routes to imitate. It learned by reinforcement: plan a route from every possible first
            stop, reward the ones shorter than their siblings, repeat over millions of routes.
          </p>
          <p>
            At inference it uses the same trick on 8 rotated or mirrored copies of the map: about 1,200 different
            candidate routes in a single pass on the GPU.
          </p>
        </article>
        <article className="why">
          <p className="big">
            {pct(s.network.gap_ctrl_pct, 1)} → {pct(s.ours_ils2.gap_ctrl_pct, 2)}
          </p>
          <h3>A neural start, a classic finish</h3>
          <p>
            The network is fast but not exact; local search is exact but needs a good start. Polishing the 8 best
            candidates brings it to {pct(s.ours.gap_ctrl_pct, 2)} in half a second, and keeping on searching to{" "}
            {pct(s.ours_ils2.gap_ctrl_pct, 2)} in 2 seconds.
          </p>
          <p>
            With the same time, PyVRP is at {pct(s.pyvrp_2s.gap_ctrl_pct, 2)} and OR-Tools at{" "}
            {pct(s.ortools_2s.gap_ctrl_pct, 2)}.
          </p>
        </article>
        <article className="why">
          <p className="big">0.5 s</p>
          <h3>Engineered to be cheap</h3>
          <p>
            Profiling showed the bottleneck was not the network but scoring candidates in Python; vectorising it made
            the whole pipeline 2.5× faster with identical answers. The local search is compiled with Numba.
          </p>
          <p>
            The result plans a 150-stop route in half a second on a 2019 GTX 1660 Super, and in about two seconds on a
            CPU, with no solver licences.
          </p>
        </article>
      </div>

      <h3>On whole station-days</h3>
      <Summary days={index.days} />

      <h3>All results on {index.headline.routes} unseen routes</h3>
      <ResultsTable index={index} />
    </section>
  );
}
