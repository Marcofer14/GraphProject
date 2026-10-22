import type { Map as MLMap } from "maplibre-gl";
import { useMemo, useState } from "react";

import { ROUTE_CHOICES, SOLVERS } from "../data";
import { hours, seconds } from "../format";
import { endTime, tracks } from "../playback";
import type { Listing, RouteDoc } from "../types";
import { usePlayback, useReplay } from "../usePlayback";
import { useSyncedCameras } from "../useSyncedCameras";
import { ReplayControls } from "./ReplayControls";
import { RouteMap } from "./RouteMap";
import { RoutePicker } from "./RoutePicker";

const NEURAL = new Set(["ours", "network"]);

interface PanelProps {
  doc: RouteDoc;
  solver: string;
  other: string;
  onSolver: (key: string) => void;
  time: number;
  dark: boolean;
  onMap: (map: MLMap | null) => void;
}

function Panel({ doc, solver, other, onSolver, time, dark, onMap }: PanelProps) {
  const sol = doc.solvers[solver];
  const ref = doc.solvers[other];
  const meta = SOLVERS[solver];
  const delta = sol.travel_h - ref.travel_h;
  const deltaText =
    Math.abs(delta) < 1 / 120
      ? "same as the other side"
      : `${delta < 0 ? "−" : "+"}${Math.round(Math.abs(delta) * 60)} min vs ${SOLVERS[other].name.toLowerCase()}`;
  return (
    <section className="panel" aria-label={meta.name}>
      <label className="panel-head">
        <span className="dot" style={{ background: meta.color }} aria-hidden="true" />
        <select value={solver} onChange={(e) => onSolver(e.target.value)}>
          {ROUTE_CHOICES.map((k) => (
            <option key={k} value={k}>
              {SOLVERS[k].name}
              {k === "ortools" || k === "pyvrp" ? " (10 s)" : ""}
            </option>
          ))}
        </select>
      </label>
      <RouteMap
        depot={doc.depot}
        stops={doc.stops}
        solution={sol}
        color={meta.color}
        time={time}
        dark={dark}
        onMap={onMap}
      />
      <dl className="stats">
        <div>
          <dt>Driving</dt>
          <dd>{hours(sol.travel_h)}</dd>
        </div>
        <div>
          <dt>Difference</dt>
          <dd className={delta < -1 / 120 ? "better" : delta > 1 / 120 ? "worse" : ""}>{deltaText}</dd>
        </div>
        <div>
          <dt>Late stops</dt>
          <dd>{sol.late}</dd>
        </div>
        <div>
          <dt>Compute</dt>
          <dd>
            {seconds(sol.runtime_s)}
            {NEURAL.has(solver) ? " (GPU)" : solver === "driver" ? "" : " (CPU)"}
          </dd>
        </div>
      </dl>
    </section>
  );
}

interface Props {
  routes: Listing[];
  routeId: string | undefined;
  onRoute: (id: string) => void;
  doc: RouteDoc | null;
  dark: boolean;
  replay?: string; // id of the route or day to replay from the start once loaded (the chat's tour)
  onReplayed?: () => void;
}

export function RouteCompare({ routes, routeId, onRoute, doc, dark, replay, onReplayed }: Props) {
  const [left, setLeft] = useState("driver");
  const [right, setRight] = useState("ours");
  const [mapA, setMapA] = useState<MLMap | null>(null);
  const [mapB, setMapB] = useState<MLMap | null>(null);
  useSyncedCameras([mapA, mapB]);

  const end = useMemo(() => {
    if (!doc) return 0;
    return Math.max(...[left, right].map((k) => endTime(tracks(doc.depot, doc.stops, doc.solvers[k]))));
  }, [doc, left, right]);
  const { time, playing, toggle, seek, restart } = usePlayback(end, doc);
  useReplay(
    replay,
    doc,
    () => {
      setLeft("driver");
      setRight("ours");
      restart();
    },
    onReplayed,
  );

  return (
    <div id="compare">
      <div className="controls">
        <RoutePicker routes={routes} value={routeId} onChange={onRoute} />
        <ReplayControls
          end={end}
          time={time}
          playing={playing}
          disabled={!doc}
          onToggle={toggle}
          onSeek={seek}
        />
      </div>
      {doc && (
        <div className="panels">
          <Panel doc={doc} solver={left} other={right} onSolver={setLeft} time={time} dark={dark} onMap={setMapA} />
          <Panel doc={doc} solver={right} other={left} onSolver={setRight} time={time} dark={dark} onMap={setMapB} />
        </div>
      )}
      <p className="note">
        Lines join the stops in visiting order; they are not street geometry. Driving times come from Amazon's real,
        asymmetric travel-time matrix, and the van waits when it arrives before a delivery window opens. Our model runs
        on a GTX 1660 Super GPU; OR-Tools and PyVRP run on CPU with a 10-second budget.
      </p>
    </div>
  );
}
