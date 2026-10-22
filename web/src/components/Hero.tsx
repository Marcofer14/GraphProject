import { SOLVERS } from "../data";
import { hours, pct, seconds } from "../format";
import type { DemoIndex } from "../types";

const REPO = "https://github.com/Marcofer14/GraphProject";

export function Hero({ index }: { index: DemoIndex }) {
  const { routes: n, solvers: s } = index.headline;
  const { ours, driver, ortools, ortools_fast, pyvrp_fast } = s;
  const lessDriving = 100 * (1 - ours.travel_h / driver.travel_h);
  return (
    <header className="hero">
      <p className="eyebrow">Ruta Norte Logistics · portfolio project</p>
      <h1>Neural last-mile routing</h1>
      <p className="lead">
        A Transformer trained with reinforcement learning plans delivery routes on real Amazon data. It reads the real
        street travel times between stops, proposes many routes at once and polishes the best ones with local search.
        Below, it faces real drivers and two classical solvers on {n} routes it never saw.
      </p>
      <nav className="links">
        <a href={REPO}>Code on GitHub</a>
        <a href="#compare">Compare routes</a>
        <a href="#results">All results</a>
      </nav>

      <div className="cards">
        <article className="card">
          <p className="big">−{lessDriving.toFixed(1)} %</p>
          <p className="card-title">driving time vs real drivers</p>
          <p className="card-text">
            {hours(driver.travel_h)} → {hours(ours.travel_h)} per route, shorter on {driver.ours_wins} of {n} routes,
            with almost no late deliveries ({pct(ours.late_pct, 2)} of stops).
          </p>
        </article>
        <article className="card">
          <p className="big">
            {ortools_fast.ours_wins} / {n}
          </p>
          <p className="card-title">routes better than OR-Tools at equal time</p>
          <p className="card-text">
            Both given about half a second. Even with 10 s for OR-Tools, our model is better on {ortools.ours_wins} of{" "}
            {n} routes.
          </p>
        </article>
        <article className="card">
          <p className="big">
            {pyvrp_fast.ours_wins} / {n}
          </p>
          <p className="card-title">routes better than PyVRP at equal time</p>
          <p className="card-text">
            On par with the strongest open-source solver, but not ahead: PyVRP's average is still closer to the best
            route ({pct(pyvrp_fast.gap_ctrl_pct, 2)} vs {pct(ours.gap_ctrl_pct, 2)} extra driving).
          </p>
        </article>
      </div>
    </header>
  );
}

const ORDER = ["ours", "pyvrp_fast", "pyvrp_1s", "pyvrp", "ortools_fast", "ortools", "network", "driver"];

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
              <th scope="col">Our model better on</th>
            </tr>
          </thead>
          <tbody>
            {ORDER.filter((k) => solvers[k]).map((k) => {
              const r = solvers[k];
              const meta = SOLVERS[k];
              return (
                <tr key={k} className={k === "ours" ? "ours" : undefined}>
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
        * Extra driving above the best route any solver found for that delivery, leaving out the trip to and from the
        station that every plan must make (about three quarters of each route). Compute time is the median; our model
        runs on a GTX 1660 Super GPU (median 1.8 s on CPU), the others on CPU.
      </p>
    </section>
  );
}
