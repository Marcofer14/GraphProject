import { SOLVERS } from "../data";
import { hours, pct, seconds } from "../format";
import type { DemoIndex } from "../types";

const REPO = "https://github.com/Marcofer14/GraphProject";

export function Hero({ index }: { index: DemoIndex }) {
  const { routes: n, solvers: s } = index.headline;
  const { ours, driver, ortools, ortools_1s, pyvrp, pyvrp_1s, ours_ils1 } = s;
  const lessDriving = 100 * (1 - ours.travel_h / driver.travel_h);
  return (
    <header className="hero" id="intro">
      <p className="eyebrow">Ruta Norte Logistics · portfolio project by Marco Fernández</p>
      <h1>Neural last-mile routing</h1>
      <p className="lead lead-big">
        Every morning a delivery van leaves its station with about 150 stops to visit, and the order of those stops
        decides how long the driver spends on the road. This project is an AI that plans that order: a neural network
        that learned routing by trial and error, then polishes its plan with classic optimisation.
      </p>
      <p className="lead">
        It was trained and tested on 9,164 real Amazon delivery routes (the {n} it is scored on were never used in
        training), with the real
        driving times between stops, and compared with what the real drivers did and with two well-known route
        optimisers, Google OR-Tools and PyVRP. Everything, training included, ran on a GTX 1660 Super, a mid-range
        gaming card from 2019: planning a route takes about half a second.
      </p>
      <nav className="links">
        <a href="#how-to">How to use this page</a>
        <a href="#example">A guided example</a>
        <a href="#architecture">How it is built</a>
        <a href={REPO}>Code on GitHub</a>
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
