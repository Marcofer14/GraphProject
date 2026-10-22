import { pct } from "../format";
import type { DemoIndex } from "../types";

const REPO = "https://github.com/Marcofer14/GraphProject";
const DECISIONS = `${REPO}/blob/main/docs/decisions.md`;

const STEPS = [
  {
    title: "Real stops and travel times",
    text:
      "Each stop's position, parcel volume and delivery window, plus Amazon's real one-way driving time between " +
      "every pair of stops, which is often not the same in both directions.",
  },
  {
    title: "Encoder: every stop looks at every other",
    text:
      "A Transformer (6 layers, 8 attention heads, 1.300a0M parameters) builds a representation of each stop. " +
      "Its attention is biased by how long it really takes to drive between stops, not by straight-line distance.",
  },
  {
    title: "Decoder: hundreds of routes at once",
    text:
      "Starting from every stop, and on 8 rotated or mirrored copies of the map, it picks the next stop again and " +
      "again until the van is back at the station: about a thousand candidate routes in one GPU pass.",
  },
  {
    title: "Pick and polish",
    text:
      "The 8 candidates with the least real driving time get local search: reversing stretches (2-opt), moving " +
      "runs of stops (or-opt) and swapping stops. This is the first answer, in about half a second.",
  },
  {
    title: "Keep improving until time is up",
    text:
      "With a longer budget, it shakes up part of the best route, polishes it again and keeps the result only if " +
      "it is shorter (iterated local search), until the time runs out.",
  },
];

export function HowItWorks({ index }: { index: DemoIndex }) {
  const s = index.headline.solvers;
  return (
    <section className="block" id="how">
      <h2>How it works</h2>
      <p className="lead">
        A neural network proposes routes and classical local search refines them: the network supplies a strong
        starting point fast, the search makes it precise.
      </p>
      <ol className="pipeline">
        {STEPS.map((step, i) => (
          <li key={step.title} className="step">
            <span className="step-n" aria-hidden="true">
              {i + 1}
            </span>
            <p className="card-title">{step.title}</p>
            <p className="card-text">{step.text}</p>
          </li>
        ))}
      </ol>

      <div className="cards how-cards">
        <article className="card">
          <p className="card-title">How it learned</p>
          <p className="card-text">
            No one showed it good routes. It was trained with reinforcement learning: it plans many routes for the same
            stops and is pushed towards the ones shorter than its own average (REINFORCE with the shared POMO
            baseline). A curriculum took it from 50-stop to 100-stop synthetic routes calibrated on Amazon's data and
            then to 60 epochs on slices of real routes with their real travel times, all on one 6 GB GTX 1660 Super.
          </p>
        </article>
        <article className="card">
          <p className="card-title">A whole day</p>
          <p className="card-text">
            For a station's whole day, the stops are first split into one compact group per van (k-means that respects
            van capacity), each group is routed as above, and a final local search moves stops between vans when that
            saves time.
          </p>
        </article>
      </div>

      <h3>Limits, honestly</h3>
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
        <li>
          The network runs on a GPU. On a CPU the first answer takes about 2 s instead of half a second.
        </li>
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
