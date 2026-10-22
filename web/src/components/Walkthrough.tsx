import { useEffect, useMemo, useState } from "react";

import { loadRoute } from "../data";
import { hours, pct } from "../format";
import { bestTravel, raceSeries, resultAt } from "../race";
import type { DemoIndex, RouteDoc } from "../types";

// the demo route whose saving against the driver is closest to the 200-route average
const EXAMPLE = "RouteID_c9d4386d-f351-45ab-8a62-b359a77f1190";

interface Props {
  index: DemoIndex;
  onRoute: (id: string) => void;
  onBudget: (s: number) => void;
  onDay: (id: string) => void;
}

const goTo = (id: string) =>
  setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);

export function Walkthrough({ index, onRoute, onBudget, onDay }: Props) {
  const [doc, setDoc] = useState<RouteDoc | null>(null);
  useEffect(() => {
    loadRoute(EXAMPLE).then(setDoc, () => setDoc(null));
  }, []);

  const race = useMemo(() => {
    if (!doc) return null;
    const best = bestTravel(doc);
    const at = (key: string, budget: number) => {
      const s = raceSeries(doc, "gpu").find((x) => x.key === key)!;
      const r = resultAt(s, budget);
      return r ? Math.round((r.sol.travel_h - best) * 60) : null;
    };
    return { ours1: at("ours", 1), pyvrp1: at("pyvrp", 1), ortools1: at("ortools", 1), pyvrp10: at("pyvrp", 10) };
  }, [doc]);

  const day = useMemo(() => [...index.days].sort((a, b) => b.stops - a.stops)[0], [index]);
  const s = index.headline.solvers;
  const mins = (m: number | null) => (m === null ? "—" : m === 0 ? "the best route found" : `${m} min above the best`);

  if (!doc || !race) return null;
  const driver = doc.solvers.driver;
  const ours = doc.solvers.ours;
  const saving = 100 * (1 - ours.travel_h / driver.travel_h);

  return (
    <section className="block" id="example">
      <h2>A guided example</h2>
      <p className="lead">
        Four steps on real data. Each “Show me” button sets up the tool below and scrolls to it; come back here for the
        next step.
      </p>
      <ol className="walk">
        <li className="walk-step">
          <div>
            <p className="card-title">Open a typical route</p>
            <p className="card-text">
              A {doc.city} route from {doc.date}, {doc.stops.length} stops. Of the 25 routes on this page, its saving is
              the closest to the average over all 200 test routes, so it is a typical case, not the best one. The driver
              drove {hours(driver.travel_h)}; our model's plan needs {hours(ours.travel_h)}, {pct(saving)} less, and
              took {ours.runtime_s.toFixed(1)} s to compute.
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              onRoute(EXAMPLE);
              goTo("compare");
            }}
          >
            Show me
          </button>
        </li>
        <li className="walk-step">
          <div>
            <p className="card-title">Replay the shift</p>
            <p className="card-text">
              In “Compare”, press “Replay the shift”. Both vans leave the station together and visit exactly the same
              stops; only the order differs. Watch the clock: our van is back at the station earlier.
            </p>
          </div>
          <button type="button" onClick={() => goTo("compare")}>
            Take me there
          </button>
        </li>
        <li className="walk-step">
          <div>
            <p className="card-title">Give every solver one second</p>
            <p className="card-text">
              With one second of computing on this route, our model returns {mins(race.ours1)}, PyVRP{" "}
              {mins(race.pyvrp1)} and OR-Tools {mins(race.ortools1)}. Given 10 s, PyVRP reaches {mins(race.pyvrp10)}.
              On this route ours stays ahead; across all {index.headline.routes} test routes PyVRP with 10 s ends up
              slightly better on average ({pct(s.pyvrp.gap_ctrl_pct, 2)} vs {pct(s.ours_ils2.gap_ctrl_pct, 2)} for our
              model with 2 s).
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              onRoute(EXAMPLE);
              onBudget(1);
              goTo("race");
            }}
          >
            Show me
          </button>
        </li>
        {day && (
          <li className="walk-step">
            <div>
              <p className="card-title">Plan a whole day</p>
              <p className="card-text">
                The largest day here: {day.city}, {day.stops.toLocaleString("en")} stops. The drivers drove{" "}
                {hours(day.hours.driver)} in total; our model plans {hours(day.hours.ours)} (
                {pct(100 * (1 - day.hours.ours / day.hours.driver))} less) in a few seconds. PyVRP, given a full
                minute, ends at {hours(day.hours.pyvrp)}
                {day.hours.pyvrp > day.hours.driver ? ", even more than the drivers: a day this size is too big for it to converge in a minute." : "."}
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                onDay(day.id);
                goTo("day");
              }}
            >
              Show me
            </button>
          </li>
        )}
      </ol>
    </section>
  );
}
