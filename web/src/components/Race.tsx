import type { Map as MLMap } from "maplibre-gl";
import { useMemo, useState } from "react";

import { seconds } from "../format";
import { bestTravel, type Hardware, NO_ROUTE, raceSeries, resultAt, toPosition, toSeconds } from "../race";
import type { Listing, RouteDoc } from "../types";
import { useSyncedCameras } from "../useSyncedCameras";
import { RaceChart } from "./RaceChart";
import { RouteMap } from "./RouteMap";
import { RoutePicker } from "./RoutePicker";

const SHOWN = ["ours", "pyvrp", "ortools"]; // on the maps; the network alone is only on the chart
const PRESETS = [1, 2, 10];

interface Props {
  routes: Listing[];
  routeId: string | undefined;
  onRoute: (id: string) => void;
  doc: RouteDoc | null;
  dark: boolean;
  budget: number; // seconds of computing; lives in App so the guided example can set it
  onBudget: (s: number) => void;
}

export function Race({ routes, routeId, onRoute, doc, dark, budget, onBudget: setBudget }: Props) {
  const [hardware, setHardware] = useState<Hardware>("gpu");
  const [m0, setM0] = useState<MLMap | null>(null);
  const [m1, setM1] = useState<MLMap | null>(null);
  const [m2, setM2] = useState<MLMap | null>(null);
  useSyncedCameras([m0, m1, m2]);
  const setters = [setM0, setM1, setM2];

  const series = useMemo(() => (doc ? raceSeries(doc, hardware) : []), [doc, hardware]);
  const best = useMemo(() => (doc ? bestTravel(doc) : 0), [doc]);

  return (
    <section className="block" id="race">
      <h2>The race: quality against computing time</h2>
      <p className="lead">
        Classical solvers keep improving the longer they run, and so does our model: after its first answer it keeps
        kicking and re-polishing its best route until the time is up. Move the time budget to see the route each one
        returns after that many seconds, measured in extra minutes of driving above the best route any of them found.
      </p>
      <div className="controls">
        <RoutePicker routes={routes} value={routeId} onChange={onRoute} />
        <div className="seg" role="group" aria-label="Our model runs on">
          {(["gpu", "cpu"] as const).map((h) => (
            <button key={h} type="button" aria-pressed={hardware === h} onClick={() => setHardware(h)}>
              Our model on {h.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
      {doc && (
        <>
          <div className="chart-wrap">
            <RaceChart
              series={series}
              best={best}
              driverH={doc.solvers.driver.travel_h}
              budget={budget}
              onBudget={setBudget}
            />
          </div>
          <div className="controls">
            <input
              type="range"
              aria-label="Seconds of computing"
              min={0}
              max={1000}
              value={toPosition(budget)}
              onChange={(e) => setBudget(toSeconds(Number(e.target.value)))}
            />
            <span className="clock">{seconds(budget)} of computing</span>
            {PRESETS.map((p) => (
              <button key={p} type="button" onClick={() => setBudget(p)}>
                {p} s
              </button>
            ))}
          </div>
          <div className="race-maps">
            {SHOWN.map((key, i) => {
              const s = series.find((x) => x.key === key)!;
              const r = resultAt(s, budget);
              const extra = r ? Math.round((r.sol.travel_h - best) * 60) : null;
              return (
                <section key={key} className="panel" aria-label={s.name}>
                  <div className="panel-head">
                    <span className="dot" style={{ background: s.color }} aria-hidden="true" />
                    <strong>{s.name}</strong>
                    <span className="muted race-status">
                      {r
                        ? `${extra === 0 ? "the best route" : `+${extra} min`} · ${seconds(r.t)}`
                        : `still computing (first answer at ${seconds(s.points[0].t)})`}
                    </span>
                  </div>
                  <RouteMap
                    depot={doc.depot}
                    stops={doc.stops}
                    solution={r ? r.sol : NO_ROUTE}
                    color={s.color}
                    time={Infinity}
                    dark={dark}
                    onMap={setters[i]}
                    small
                  />
                </section>
              );
            })}
          </div>
        </>
      )}
      <p className="note">
        Every solver was run once per budget and each point is what that run returned: OR-Tools and PyVRP at 0.5, 1,
        2, 5 and 10 s on CPU; our model's first answer, then 1 and 2 s with its search, with the network on a GTX 1660
        Super GPU and the search on one CPU core. The CPU switch shows our first answer when the network also runs on
        the CPU. Over all 200 test routes with one second each, our model is closer to the best route than PyVRP on
        average and on most routes; PyVRP given 10 s is still better.
      </p>
    </section>
  );
}
