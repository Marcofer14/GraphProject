import { SOLVERS } from "../data";
import { hours, pct, seconds } from "../format";
import type { DemoIndex } from "../types";
import { openChat } from "../chat/assistant";

const REPO = "https://github.com/Marcofer14/GraphProject";

export function Hero({ index }: { index: DemoIndex }) {
  const { routes: n, solvers: s } = index.headline;
  const { ours, driver, pyvrp_1s, ortools_1s } = s;
  const lessDriving = 100 * (1 - ours.travel_h / driver.travel_h);
  return (
    <header className="hero" id="intro">
      <div className="hero-grid">
        <div>
          <p className="eyebrow">Ruta Norte Logistics · portfolio project by Marco Fernández</p>
          <h1>
            Neural last-mile <i>routing</i>
          </h1>
          <p className="lead lead-big">
            Every morning a delivery van leaves its station with about 150 stops to visit. The order of those stops
            decides how long the driver spends on the road, and with 150 stops there are more possible orders than
            atoms in the universe, so no one can try them all.
          </p>
          <p className="lead">
            This project is an AI that plans that order. A neural network learned routing by trial and error on 9,164
            real Amazon delivery routes, with the real driving times between stops, and a classic optimiser polishes
            its plan. It is compared with what the real drivers did and with two well-known route optimisers, Google
            OR-Tools and PyVRP, on {n} routes it never saw. Everything, training included, ran on a GTX 1660 Super, a
            mid-range gaming card from 2019.
          </p>
          <div className="hero-actions">
            <a className="primary" href="#demo">
              Try the demo
            </a>
            <a
              href="#help"
              onClick={(e) => {
                e.preventDefault();
                openChat();
              }}
            >
              Ask the project
            </a>
            <a href="#architecture">How it is built</a>
            <a href={REPO}>Code on GitHub ↗</a>
          </div>
        </div>
        <dl className="problem">
          <dt>The problem</dt>
          <dd>Plan the order of each van's stops so the day needs as little driving as possible, on real streets.</dd>
          <dt>The approach</dt>
          <dd>
            A Transformer trained with reinforcement learning proposes about a thousand routes at once; local search
            polishes the best ones.
          </dd>
          <dt>The result</dt>
          <dd>
            {pct(lessDriving)} less driving than the real drivers ({hours(driver.travel_h)} → {hours(ours.travel_h)}{" "}
            per route), in about half a second, and ahead of PyVRP and OR-Tools when all get the same computing time.
          </dd>
        </dl>
      </div>

      <div className="cards" style={{ marginTop: 24 }}>
        <article className="card">
          <p className="big">−{lessDriving.toFixed(1)}%</p>
          <p className="card-title">driving time vs real drivers</p>
          <p className="card-text">About 20 minutes less per van per day, shorter on all {n} test routes.</p>
        </article>
        <article className="card">
          <p className="big">
            {pyvrp_1s.ours_wins}/{n}
          </p>
          <p className="card-title">routes better than PyVRP</p>
          <p className="card-text">Both given one second of computing.</p>
        </article>
        <article className="card">
          <p className="big">
            {ortools_1s.ours_wins}/{n}
          </p>
          <p className="card-title">routes better than Google OR-Tools</p>
          <p className="card-text">Both given one second of computing.</p>
        </article>
        <article className="card">
          <p className="big">0.5 s</p>
          <p className="card-title">to plan a 150-stop route</p>
          <p className="card-text">On a 2019 mid-range gaming GPU; about 2 s on a CPU.</p>
        </article>
      </div>
    </header>
  );
}

const ORDER = [
  "pyvrp",
  "ours_ils2",
  "pyvrp_2s",
  "ours_ils1",
  "pyvrp_1s",
  "pyvrp_fast",
  "ours",
  "ortools",
  "ortools_2s",
  "ortools_1s",
  "network",
  "ortools_fast",
  "driver",
];
const OURS = new Set(["ours", "ours_ils1", "ours_ils2"]);

export function ResultsTable({ index }: { index: DemoIndex }) {
  const { routes: n, solvers } = index.headline;
  return (
    <div id="results">
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Solver</th>
              <th scope="col">Compute</th>
              <th scope="col">Driving per route</th>
              <th scope="col">Above the best route*</th>
              <th scope="col">Late stops</th>
              <th scope="col">Our model better on†</th>
            </tr>
          </thead>
          <tbody>
            {ORDER.filter((k) => solvers[k]).map((k) => {
              const r = solvers[k];
              const meta = SOLVERS[k];
              return (
                <tr key={k} className={OURS.has(k) ? "ours" : undefined}>
                  <th scope="row">
                    <span className="dot" style={{ background: meta.color }} aria-hidden="true" />
                    {meta.name}
                    {meta.note && <span className="muted"> · {meta.note}</span>}
                  </th>
                  <td>{seconds(r.runtime_s)}</td>
                  <td>{hours(r.travel_h)}</td>
                  <td>{pct(r.gap_ctrl_pct, 2)}</td>
                  <td>{pct(r.late_pct, 2)}</td>
                  <td>{r.ours_wins === undefined ? "—" : `${r.ours_wins} / ${n}`}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="note">
        * Extra driving above the best route any of these solvers found for that delivery, leaving out the trip to and
        from the station that every plan must make (about three quarters of each route). Compute time is the median.
        Our model's network runs on a GTX 1660 Super GPU (its first answer takes a median 2.1 s on CPU); its local
        search and both classical solvers use one CPU core.
      </p>
      <p className="note">
        † Compared route by route with our model given the same time; against the 10-second budgets, with our 2-second
        run; against the driver and the network alone, with its first answer.
      </p>
    </div>
  );
}
