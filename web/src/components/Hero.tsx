import { SOLVERS } from "../data";
import { hours, pct, seconds } from "../format";
import type { DemoIndex } from "../types";

const REPO = "https://github.com/Marcofer14/GraphProject";

export function Hero({ index }: { index: DemoIndex }) {
  const { routes: n, solvers: s } = index.headline;
  const { ours, driver, ortools, ortools_1s, pyvrp, pyvrp_1s, ours_ils1 } = s;
  const lessDriving = 100 * (1 - ours.travel_h / driver.travel_h);
  return (
    <header className="hero">
      <p className="eyebrow">Ruta Norte Logistics · portfolio project</p>
      <h1>Neural last-mile routing</h1>
      <p className="lead">
        A Transformer trained with reinforcement learning plans delivery routes on real Amazon data. It reads the real
        street travel times between stops, proposes many routes at once, polishes the best ones and keeps improving
        them with local search for as long as it is given. Below, it faces real drivers and two classical solvers on{" "}
        {n} routes it never saw.
      </p>
      <nav className="links">
        <a href={REPO}>Code on GitHub</a>
        <a href="#compare">Compare routes</a>
        <a href="#race">The race</a>
        <a href="#day">A whole day</a>
        <a href="#results">All results</a>
      </nav>

      <div className="cards">
        <article className="card">
          <p className="big">−{lessDriving.toFixed(1)} %</p>
          <p className="card-title">driving time vs real drivers</p>
          <p className="card-text">
            {hours(driver.travel_h)} → {hours(ours.travel_h)} per route from the first answer (half a second), shorter
            on {driver.ours_wins} of {n} routes, with almost no late deliveries.
          </p>
        </article>
        <article className="card">
          <p className="big">
            {pyvrp_1s.ours_wins} / {n}
          </p>
          <p className="card-title">routes better than PyVRP at equal time</p>
          <p className="card-text">
            Both given one second, our model is also closer to the best route on average ({pct(ours_ils1.gap_ctrl_pct, 2)}{" "}
            vs {pct(pyvrp_1s.gap_ctrl_pct, 2)} extra driving). PyVRP given 10 s is still better (
            {pct(pyvrp.gap_ctrl_pct, 2)}).
          </p>
        </article>
        <article className="card">
          <p className="big">
            {ortools_1s.ours_wins} / {n}
          </p>
          <p className="card-title">routes better than OR-Tools at equal time</p>
          <p className="card-text">
            Both given one second. Even with 10 s for OR-Tools against 2 s for our model, ours is better on{" "}
            {ortools.ours_wins} of {n} routes.
          </p>
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
    <section className="block" id="results">
      <h2>All results on {n} unseen routes</h2>
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
    </section>
  );
}
