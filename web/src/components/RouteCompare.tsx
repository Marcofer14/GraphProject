import type { Map as MLMap } from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";

import { ROUTE_CHOICES, SOLVERS } from "../data";
import { clock, hours, seconds } from "../format";
import { endTime, tracks } from "../playback";
import type { Listing, RouteDoc } from "../types";
import { useSyncedCameras } from "../useSyncedCameras";
import { RouteMap } from "./RouteMap";
import { RoutePicker } from "./RoutePicker";

const PLAY_SECONDS = 14; // a whole shift replays in this many seconds
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
}

export function RouteCompare({ routes, routeId, onRoute, doc, dark }: Props) {
  const [left, setLeft] = useState("driver");
  const [right, setRight] = useState("ours");
  const [time, setTime] = useState(Infinity);
  const [playing, setPlaying] = useState(false);
  const [mapA, setMapA] = useState<MLMap | null>(null);
  const [mapB, setMapB] = useState<MLMap | null>(null);
  useSyncedCameras([mapA, mapB]);

  useEffect(() => {
    setTime(Infinity);
    setPlaying(false);
  }, [doc]);

  const end = useMemo(() => {
    if (!doc) return 0;
    return Math.max(...[left, right].map((k) => endTime(tracks(doc.depot, doc.stops, doc.solvers[k]))));
  }, [doc, left, right]);

  const frame = useRef(0);
  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      let done = false;
      setTime((t) => {
        const next = (Number.isFinite(t) ? t : 0) + (dt * end) / PLAY_SECONDS;
        done = next >= end;
        return done ? Infinity : next;
      });
      if (done) setPlaying(false);
      else frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
  }, [playing, end]);

  const play = () => {
    if (!Number.isFinite(time)) setTime(0);
    setPlaying((p) => !p);
  };

  return (
    <section className="block" id="compare">
      <h2>Compare on a real route</h2>
      <p className="lead">
        Pick one of {routes.length} real Amazon routes, drawn at random from the 200 the model never saw, and compare
        any two plans. Replay shows both vans leaving the station at the same time.
      </p>
      <div className="controls">
        <RoutePicker routes={routes} value={routeId} onChange={onRoute} />
        <button type="button" onClick={play} disabled={!doc}>
          {playing ? "Pause" : Number.isFinite(time) ? "Resume" : "Replay the shift"}
        </button>
        <input
          type="range"
          aria-label="Time into the shift"
          min={0}
          max={Math.ceil(end)}
          step={60}
          value={Number.isFinite(time) ? Math.round(time) : Math.ceil(end)}
          onChange={(e) => {
            setPlaying(false);
            const t = Number(e.target.value);
            setTime(t >= end ? Infinity : t);
          }}
        />
        <span className="clock" aria-live="off">
          {clock(Number.isFinite(time) ? time : end)} into the shift
        </span>
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
    </section>
  );
}
