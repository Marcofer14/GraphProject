import type { ReactNode } from "react";

import { pct } from "../format";
import type { DemoIndex } from "../types";

const REPO = "https://github.com/Marcofer14/GraphProject";
const DECISIONS = `${REPO}/blob/main/docs/decisions.md`;

/** Boxes joined by arrows: a row on wide screens, a column on phones. */
function Flow({ steps }: { steps: { title: string; text: ReactNode; tag?: string }[] }) {
  return (
    <ol className="flow">
      {steps.map((s) => (
        <li key={s.title} className="flow-step">
          {s.tag && <span className="flow-tag">{s.tag}</span>}
          <p className="card-title">{s.title}</p>
          <p className="card-text">{s.text}</p>
        </li>
      ))}
    </ol>
  );
}

function Layer({ name, detail, kind }: { name: string; detail: string; kind?: "input" | "bias" | "out" }) {
  return (
    <div className={`layer${kind ? ` layer-${kind}` : ""}`}>
      <b>{name}</b>
      <span>{detail}</span>
    </div>
  );
}

export function Architecture({ index }: { index: DemoIndex }) {
  const s = index.headline.solvers;
  return (
    <section className="block" id="architecture">
      <h2>How it is built</h2>
      <p className="lead">
        From a public dataset to this page: the data pipeline, the neural network, how it learned, how a plan is
        produced in half a second, and how every number on this page was measured.
      </p>

      <h3>1 · The system end to end</h3>
      <Flow
        steps={[
          { tag: "Data", title: "Amazon Last Mile dataset", text: "9,164 real routes and 1.34 M stops from AWS Open Data, with 1.7 GB of real driving-time matrices." },
          { tag: "ETL", title: "Streaming pipeline", text: "JSON parsed as a stream (ijson) into Parquet tables and one memory-mapped matrix file, so 1.7 GB never sits in memory." },
          { tag: "Training", title: "PyTorch on one GPU", text: "Hydra configs, TensorBoard, resumable checkpoints, early stopping; a GTX 1660 Super with 6 GB." },
          { tag: "Solvers", title: "One interface, one judge", text: "Our model, OR-Tools, PyVRP and classic heuristics behind the same API, all scored by the same evaluator." },
          { tag: "Benchmarks", title: "Unseen data only", text: "200 test routes and 20 station-days no model saw; equal-time runs on an otherwise idle machine." },
          { tag: "This page", title: "Static and fast", text: "Results exported once to JSON; React, TypeScript and MapLibre on GitHub Pages. The repository also has a FastAPI service to solve new routes on demand." },
        ]}
      />

      <h3>2 · The neural network</h3>
      <p className="prose">
        A Transformer encoder–decoder in the style of the Attention Model (Kool et al., 2019), with 1.3 M parameters.
        Its key addition is that attention is biased by the <em>real</em> one-way driving times between stops: a small
        network turns each pair's times into a score added inside every attention layer, so two stops that are close on
        the map but far apart by road are treated as far apart. Before this, the network alone was 38 % above the best
        route; with it, under 10 %.
      </p>
      <div className="model">
        <div className="model-col">
          <p className="model-h">Encoder · reads the whole problem once</p>
          <Layer kind="input" name="Inputs per stop" detail="position relative to the stops' centre; station as direction and log-distance; parcel volume ÷ van capacity" />
          <Layer kind="bias" name="Inputs per pair of stops" detail="real driving time each way, log-scaled → small MLP → one bias per attention head" />
          <Layer name="Embedding" detail="linear projection to 128 dimensions" />
          <Layer name="6 × attention block" detail="8-head self-attention (+ travel-time bias) → feed-forward 512 → instance norm" />
          <Layer kind="out" name="Stop embeddings" detail="one 128-d vector per stop, and their mean as a summary of the day" />
        </div>
        <div className="model-col">
          <p className="model-h">Decoder · picks one stop at a time</p>
          <Layer kind="input" name="Context" detail="summary of the day + current stop + first stop + van capacity left" />
          <Layer name="Glimpse" detail="8-head attention from the context over all stops (+ travel-time bias from the current stop)" />
          <Layer name="Scores" detail="compatibility with every stop + travel-time bias, clipped with 10·tanh" />
          <Layer kind="bias" name="Mask" detail="stops already visited, or too big for what is left in the van" />
          <Layer kind="out" name="Next stop" detail="highest score; back to the station when the van is full, until every stop is served" />
        </div>
      </div>

      <h3>3 · How it learned</h3>
      <p className="prose">
        Nobody showed it good routes. It learned with reinforcement learning (REINFORCE with the POMO baseline, Kwon et
        al., 2020): for each training problem it plans one route starting from every stop, and every route is nudged up
        or down by how much shorter or longer it is than the average of its siblings. Repeated over millions of routes,
        the network gets better at what shortens a route.
      </p>
      <Flow
        steps={[
          { tag: "Stage 1", title: "50 stops, synthetic", text: "Random problems from a generator calibrated on Amazon's data: a distant station, stops in clusters, parcel sizes drawn from the real ones." },
          { tag: "Stage 2", title: "100 stops, synthetic", text: "Warm-started from stage 1 on larger problems from the same generator." },
          { tag: "Stage 3", title: "100-stop slices of real routes", text: "Real stops with their real driving times; the travel-time bias is added, initialised at zero so learning starts exactly where stage 2 ended. 60 epochs, under three hours." },
        ]}
      />
      <p className="prose small">
        Validation always uses held-out data, and training stops early when the part of the route a planner can
        actually change stops improving. The 3,052 routes of the evaluation split are never used for training.
      </p>

      <h3>4 · From stops to a plan in half a second</h3>
      <Flow
        steps={[
          { tag: "GPU", title: "Encode once", text: "The whole route goes through the encoder in one pass." },
          { tag: "GPU", title: "Decode ~1,200 candidates", text: "One route from every first stop, on 8 rotated or mirrored copies of the map, all in parallel." },
          { tag: "CPU", title: "Score them exactly", text: "Every candidate is scored on the real time matrix (travel + lateness), vectorised in NumPy." },
          { tag: "CPU", title: "Polish the best 8", text: "Local search compiled with Numba: reverse stretches (2-opt, exact for one-way times), move runs of stops (or-opt), swap stops." },
          { tag: "CPU", title: "Keep improving", text: "If time is left: shake up a small part of the best route, polish again, keep it only if shorter, until the budget is spent." },
        ]}
      />
      <p className="prose small">
        Profiling showed the network was not the bottleneck: scoring candidates one by one in Python was. Vectorising it
        and checking “are all routes finished?” every 8 steps instead of every step made inference 2.5× faster with
        identical results.
      </p>

      <h3>5 · A whole station-day</h3>
      <Flow
        steps={[
          { title: "Split into vans", text: "Capacity-aware k-means: k-means++ starts, stops assigned in order of regret, never over a van's volume." },
          { title: "Route each van", text: "Each group is a one-van problem for the neural pipeline above, on its own part of the time matrix." },
          { title: "Trade stops", text: "A final local search moves and swaps stops between vans when it saves time." },
        ]}
      />

      <h3>6 · How the numbers were measured</h3>
      <ul className="limits measure">
        <li>
          <b>One judge.</b> Every plan, ours or not, is scored by the same evaluator: driving time on Amazon's real
          matrix, deliveries after their window counted as late, van capacity checked.
        </li>
        <li>
          <b>A fair yardstick.</b> About three quarters of every route is the unavoidable trip to and from the station;
          results are reported as extra driving above the best route found, on the part a planner can change.
        </li>
        <li>
          <b>Equal time, no load.</b> Solvers are compared with the same computing budget, run on an otherwise idle
          machine: a one-second budget is sensitive to anything else running. With 2 s ours is{" "}
          {pct(s.ours_ils2.gap_ctrl_pct, 2)} above the best on average, PyVRP {pct(s.pyvrp_2s.gap_ctrl_pct, 2)}.
        </li>
        <li>
          <b>Engineering.</b> 129 automated tests, strict type checking (mypy) and linting (Ruff) on every push with
          GitHub Actions, which also deploys this page; 21 design decisions written up with their numbers in the{" "}
          <a href={DECISIONS}>repository</a>.
        </li>
      </ul>
    </section>
  );
}
