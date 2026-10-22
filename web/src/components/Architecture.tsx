import type { ReactNode } from "react";

import { pct } from "../format";
import type { DemoIndex } from "../types";

const REPO = "https://github.com/Marcofer14/GraphProject";
const DECISIONS = `${REPO}/blob/main/docs/decisions.md`;

interface Item {
  tag?: string;
  title: string;
  text: ReactNode;
  formula?: string;
}

/** Boxes joined by arrows: a row on wide screens, a column on narrow ones. */
function Flow({ steps }: { steps: Item[] }) {
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

/** A grid of explanatory boxes, optionally with a formula. */
function Boxes({ items }: { items: Item[] }) {
  return (
    <div className="boxes">
      {items.map((b) => (
        <article key={b.title} className="box">
          {b.tag && <span className="flow-tag">{b.tag}</span>}
          <p className="card-title">{b.title}</p>
          <p>{b.text}</p>
          {b.formula && <code className="formula">{b.formula}</code>}
        </article>
      ))}
    </div>
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

function Part({ n, title, intro, children }: { n: number; title: string; intro: ReactNode; children: ReactNode }) {
  return (
    <div className="arch-part">
      <p className="eyebrow">Part {n} of 8</p>
      <h3>{title}</h3>
      <p className="prose">{intro}</p>
      {children}
    </div>
  );
}

export function Architecture({ index }: { index: DemoIndex }) {
  const s = index.headline.solvers;
  return (
    <section className="block" id="architecture">
      <div className="sec-head">
        <h2>
          How it is <i>built</i>
        </h2>
        <span className="num">05</span>
      </div>
      <p className="lead">
        The whole system, from a public dataset to this page, in eight parts: the pipeline, the data and how results
        are judged, how stops become numbers, the neural network, how it learned, how a plan is made in half a second,
        how whole days are planned, and the software around it. Every part was measured; the reasoning behind each
        choice is written up in 21 <a href={DECISIONS}>design decisions</a>.
      </p>

      <Part
        n={1}
        title="The system end to end"
        intro="Six stages, each a separate, tested piece of a Python package, connected by files on disk so any stage can be re-run on its own."
      >
        <Flow
          steps={[
            { tag: "Data", title: "Amazon Last Mile dataset", text: "9,164 real routes, 1.34 M stops and 1.7 GB of real driving-time matrices from AWS Open Data." },
            { tag: "ETL", title: "Streaming pipeline", text: "Parsed as a stream into Parquet tables and one memory-mapped matrix file." },
            { tag: "Training", title: "PyTorch on one GPU", text: "Reinforcement learning with a three-stage curriculum on a GTX 1660 Super." },
            { tag: "Solvers", title: "One interface", text: "Our model, OR-Tools, PyVRP and classic heuristics behind the same API." },
            { tag: "Benchmarks", title: "One judge", text: "Every plan scored by the same evaluator, on routes and days never used in training." },
            { tag: "Delivery", title: "This page", text: "Results exported to static JSON; React and MapLibre on GitHub Pages; FastAPI for new routes." },
          ]}
        />
      </Part>

      <Part
        n={2}
        title="The data and how results are judged"
        intro="Real data, a strict split between what the model learns from and what it is tested on, and one evaluator that scores every plan the same way, the drivers' included."
      >
        <Boxes
          items={[
            {
              tag: "Data",
              title: "Real stops, parcels and windows",
              text: "Each route has its station, stops with GPS positions, every parcel's size and delivery window, the van's capacity, and the order the driver actually drove. Five US metro areas, 2018.",
            },
            {
              tag: "Data",
              title: "Real one-way driving times",
              text: "For every pair of stops of a route, the driving time from Amazon's data. A→B often differs from B→A (one-way streets, turns), so the problem is asymmetric: methods that assume symmetric distances are wrong here.",
            },
            {
              tag: "ETL",
              title: "Streaming, memory-safe pipeline",
              text: "The 1.7 GB matrix file is parsed as a stream (ijson) and written to one float32 file that is memory-mapped: a route's matrix is read straight from disk when needed, so nothing large ever sits in memory.",
            },
            {
              tag: "Split",
              title: "Never tested on what it learned",
              text: "6,112 routes train the model (10% of them, chosen by a hash of the route ID, held out for validation); the 3,052 routes of the evaluation split are only used for the final tests, including the 20 station-days.",
            },
            {
              tag: "Judge",
              title: "One evaluator for every plan",
              text: "Driving time is summed on the real matrix; a van that arrives early waits, one that arrives after the window closes is counted as late; van capacity is checked. Our plans, OR-Tools', PyVRP's and the drivers' all go through it.",
            },
            {
              tag: "Judge",
              title: "A fair yardstick: the controllable gap",
              text: "About three quarters of a route is the unavoidable trip from the station to the area and back. Comparing whole routes would hide real differences, so results count only the part a planner can change.",
              formula: "fixed = shortest trip out + shortest trip back\ngap = (route − fixed) / (best route − fixed) − 1",
            },
          ]}
        />
      </Part>

      <Part
        n={3}
        title="Turning stops into numbers"
        intro="A neural network needs inputs on a stable scale. How the stops are represented turned out to matter as much as the network itself."
      >
        <Boxes
          items={[
            {
              tag: "Positions",
              title: "Centred on the stops, not the station",
              text: "Coordinates are centred and scaled on the delivery area. The station, often far outside it, is given as a direction and a log-distance. Before this change, a distant station squeezed all the stops into a tiny corner of the input space and the network failed on real routes.",
            },
            {
              tag: "Parcels",
              title: "Load as a share of the van",
              text: "Each stop's parcel volume is divided by the van's capacity, so the same network works for any van size.",
            },
            {
              tag: "Times",
              title: "Travel times, both directions",
              text: "For every pair of stops the network sees log(1 + time) in each direction, which tames the range from seconds to an hour.",
              formula: "pair features (i, j) = [log1p(tᵢⱼ), log1p(tⱼᵢ)]",
            },
            {
              tag: "Synthetic data",
              title: "A generator calibrated on Amazon",
              text: "The first two training stages use random problems whose statistics were fitted on the real routes: station distance, stop clusters, parcel sizes (log-normal fits). Unlimited fresh training data, already realistic.",
            },
          ]}
        />
      </Part>

      <Part
        n={4}
        title="The neural network"
        intro={
          <>
            A Transformer encoder–decoder in the style of the Attention Model (Kool et al., 2019): 1.3 million parameters,
            128-dimensional embeddings, 6 encoder layers, 8 attention heads. The encoder reads the whole route once; the
            decoder then picks the next stop again and again until every stop is served. Its key addition is that{" "}
            <b>every attention score gets a bias computed from the real driving times</b>, in the spirit of MatNet.
          </>
        }
      >
        <div className="model">
          <div className="model-col">
            <p className="model-h">Encoder · reads the whole route once</p>
            <Layer kind="input" name="Inputs per stop" detail="centred position; station as direction and log-distance; parcel volume ÷ van capacity" />
            <Layer kind="bias" name="Inputs per pair of stops" detail="log driving time each way → small MLP → one bias per attention head" />
            <Layer name="Embedding" detail="linear projection to 128 dimensions" />
            <Layer name="6 × attention block" detail="8-head self-attention (+ travel-time bias) → feed-forward 512 → instance norm" />
            <Layer kind="out" name="Stop embeddings" detail="one 128-d vector per stop, and their mean as a summary of the route" />
          </div>
          <div className="model-col">
            <p className="model-h">Decoder · picks one stop at a time</p>
            <Layer kind="input" name="Context" detail="summary of the route + current stop + first stop + van capacity left" />
            <Layer name="Glimpse" detail="8-head attention from the context over all stops (+ travel-time bias from the current stop)" />
            <Layer name="Scores" detail="compatibility with every stop + travel-time bias, clipped with 10·tanh" />
            <Layer kind="bias" name="Mask" detail="stops already visited, or too big for what is left in the van" />
            <Layer kind="out" name="Next stop" detail="the best score; back to the station only when needed, until all stops are served" />
          </div>
        </div>
        <p className="legend">
          <span style={{ ["--c" as string]: "#888780" }}>inputs</span>
          <span style={{ ["--c" as string]: "#ba7517" }}>driving-time information</span>
          <span style={{ ["--c" as string]: "#0e7490" }}>learned layers</span>
          <span style={{ ["--c" as string]: "#2f7d5b" }}>outputs</span>
        </p>
        <Boxes
          items={[
            {
              tag: "Key idea",
              title: "Attention that knows the road",
              text: "In plain attention, how much stop i attends to stop j depends only on their embeddings. Here a learned bias from the real times between them is added, separately for each head, in every encoder layer, in the decoder's glimpse and in the final scores.",
              formula: "score(i, j) = qᵢ·kⱼ / √d + b_head(tᵢⱼ, tⱼᵢ)",
            },
            {
              tag: "Decoding",
              title: "One step of the decoder",
              text: "The context asks “where next?”; the answer is a probability over the stops still allowed. Clipping with 10·tanh keeps the scores in a range where the policy keeps exploring during training.",
              formula: "p(next = j) = softmax(10·tanh(q·kⱼ/√d + b(t_cur,j)))  over allowed j",
            },
            {
              tag: "Training trick",
              title: "Biases that start at zero",
              text: "The travel-time layers were added to a network already trained on coordinates, with their last layer initialised at zero. Training resumed exactly where the old model was and learned how much to trust the real times.",
            },
            {
              tag: "Measured",
              title: "Why it mattered",
              text: `Only 67% of each stop's five nearest neighbours by straight line are among its five nearest by road, and real legs take about 2.6× the straight-line estimate. With the bias, the network alone went from 38% to ${pct(s.network.gap_ctrl_pct, 1)} above the best route.`,
            },
          ]}
        />
      </Part>

      <Part
        n={5}
        title="How it learned"
        intro="Reinforcement learning: no example solutions, only the length of the routes it plans itself. The network is the policy; a shorter route is a better reward."
      >
        <Boxes
          items={[
            {
              tag: "Objective",
              title: "Make routes shorter",
              text: "For each training problem the network plans routes; the reward is minus their length. Gradients push up the probability of the decisions that led to shorter routes.",
            },
            {
              tag: "POMO",
              title: "Every route compared with its siblings",
              text: "It plans one route from every possible first stop (POMO, Kwon et al., 2020). Each route's advantage is how much shorter it is than the average of those N routes: a baseline that needs no second network and makes learning stable.",
              formula: "∇J ≈ 1/N · Σᵢ (Lᵢ − mean L) · ∇ log p(routeᵢ)",
            },
            {
              tag: "Optimiser",
              title: "Steady, resumable training",
              text: "Adam with gradient clipping, batches of 32 problems × 100 starting stops, Hydra configs, TensorBoard curves, checkpoints that resume exactly (optimizer included), and early stopping on the controllable gap of held-out routes.",
            },
          ]}
        />
        <Flow
          steps={[
            { tag: "Stage 1", title: "50 stops, synthetic", text: "Learn routing from scratch on the calibrated generator." },
            { tag: "Stage 2", title: "100 stops, synthetic", text: "Warm start from stage 1 on larger problems." },
            { tag: "Stage 3", title: "Real route slices", text: "100-stop slices of real routes with real times; the travel-time bias is added; 60 epochs, under three hours." },
          ]}
        />
      </Part>

      <Part
        n={6}
        title="From stops to a plan in half a second"
        intro="The network gives strong candidates fast; exact classic search makes the best one precise and keeps improving it while time is left. The GPU does the network, one CPU core does the rest."
      >
        <Flow
          steps={[
            { tag: "GPU", title: "Encode once", text: "The whole route goes through the encoder in one pass." },
            { tag: "GPU", title: "~1,200 candidates", text: "One route from every first stop, on 8 rotated or mirrored copies of the map, decoded in parallel." },
            { tag: "CPU", title: "Exact scoring", text: "All candidates scored on the real matrix (travel + 10 × lateness), vectorised in NumPy." },
            { tag: "CPU", title: "Polish the best 8", text: "Local search compiled with Numba until no move improves." },
            { tag: "CPU", title: "Keep improving", text: "Iterated local search until the time budget is spent." },
          ]}
        />
        <Boxes
          items={[
            {
              tag: "Augmentation",
              title: "8 views of the same map",
              text: "Rotating and mirroring the coordinates does not change the problem but changes what the network sees, so each view gives different candidates. Free diversity.",
            },
            {
              tag: "Local search",
              title: "2-opt, exact on one-way streets",
              text: "Reversing a stretch of a route changes the direction of every leg inside it. With one-way times that cost is computed exactly in constant time using prefix sums of forward and backward legs.",
              formula: "Δ = cost(reversed stretch) − cost(stretch), from prefix sums",
            },
            {
              tag: "Local search",
              title: "Or-opt and swap",
              text: "Move a run of 1 to 3 consecutive stops elsewhere (also reversed), or swap two stops, whenever it shortens the route and the van still has room.",
            },
            {
              tag: "Anytime",
              title: "Iterated local search",
              text: `With time left, it cuts the best route in three places inside a 50-stop window, reconnects the pieces in another order (a “double bridge” that local search cannot undo in one move), polishes again and keeps the result only if shorter. That takes it from ${pct(s.ours.gap_ctrl_pct, 2)} to ${pct(s.ours_ils2.gap_ctrl_pct, 2)} in 2 seconds.`,
            },
            {
              tag: "Speed",
              title: "Measure, then optimise",
              text: "Profiling showed 51% of the time went into scoring candidates one by one in Python, not into the network. Vectorising that and checking “are all routes done?” every 8 steps made it 2.5× faster with identical answers.",
            },
          ]}
        />
      </Part>

      <Part
        n={7}
        title="Planning a whole station-day"
        intro="A station's day has several hundred to a thousand stops and several vans. The network was trained on one van at a time, so the day is decomposed."
      >
        <Flow
          steps={[
            { tag: "Split", title: "One group per van", text: "Capacity-aware k-means: k-means++ starts, stops assigned in order of regret, never over a van's volume." },
            { tag: "Route", title: "Each van with the network", text: "Every group is a one-van problem for the pipeline above, on its own part of the time matrix." },
            { tag: "Trade", title: "Move stops between vans", text: "A final local search moves and swaps stops across vans when it saves time." },
          ]}
        />
        <Boxes
          items={[
            {
              tag: "Data gap",
              title: "Times between different routes",
              text: "Amazon only publishes driving times within each original route. Between stops of different routes the time is estimated from distance, fitted on the published times; every plan is scored with that same matrix.",
              formula: "time ≈ 194 s + 99 s per km",
            },
            {
              tag: "Result",
              title: "Fewer fleet hours in seconds",
              text: "On 20 unseen days: 9.9% fewer fleet hours than the drivers, in a few seconds. PyVRP, given a full minute, wins the smaller days but cannot converge on the larger ones.",
            },
          ]}
        />
      </Part>

      <Part
        n={8}
        title="The software around it"
        intro="Research code that runs reliably: one package, typed and tested, with every result reproducible from the command line."
      >
        <Boxes
          items={[
            {
              tag: "Package",
              title: "One Python package, five commands",
              text: "Data (nr-etl), training (nr-train), benchmarks (nr-bench), demo export (nr-demo-export) and an HTTP API (nr-api, FastAPI). Every solver registers behind the same interface, so adding one is a few lines.",
            },
            {
              tag: "Quality",
              title: "Tested and typed",
              text: "129 automated tests (pytest), strict type checking (mypy), linting and formatting (Ruff), pre-commit hooks, and GitHub Actions running all of it on every push.",
            },
            {
              tag: "Benchmarks",
              title: "Fair, parallel benchmarking",
              text: "Classic solvers run in parallel worker processes while the GPU runs the network. Equal-time comparisons were re-run on an idle machine after finding that a 1-second budget is sensitive to load.",
            },
            {
              tag: "Web",
              title: "A static, instant demo",
              text: "Every plan is solved once and exported to JSON; the page (React, TypeScript, Vite, MapLibre) only draws. A GitHub Actions workflow builds and publishes it to GitHub Pages on every change.",
            },
            {
              tag: "Assistant",
              title: "A chat that saves tokens",
              text: "Common questions are answered by scripted replies that also operate the demo. Free-form questions go to a Cloudflare Worker that picks the relevant parts of a knowledge base and asks a free open model.",
            },
            {
              tag: "Hardware",
              title: "All on a 2019 gaming GPU",
              text: "Training, benchmarks and this demo's data were all produced on one GTX 1660 Super with 6 GB: no cloud GPUs.",
            },
          ]}
        />
      </Part>
    </section>
  );
}
