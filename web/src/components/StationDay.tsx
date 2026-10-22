import type { Map as MLMap } from "maplibre-gl";
import { useMemo, useState } from "react";

import { DAY_SOLVERS, VAN_COLORS } from "../data";
import { hours, pct, seconds } from "../format";
import { alignedColors, endTime, tracks } from "../playback";
import type { DayDoc, Listing } from "../types";
import { usePlayback, useReplay } from "../usePlayback";
import { useSyncedCameras } from "../useSyncedCameras";
import { ReplayControls } from "./ReplayControls";
import { RouteMap } from "./RouteMap";
import { RoutePicker } from "./RoutePicker";

const label = (key: string) => {
  const m = DAY_SOLVERS[key];
  return m.note ? `${m.name} (${m.note})` : m.name;
};

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};

/** Averages over every exported day, so the claim covers all of them and not the one on screen. */
export function Summary({ days }: { days: Listing[] }) {
  const mean = (k: string) => days.reduce((a, d) => a + d.hours[k], 0) / days.length;
  const oursBetter = (k: string) => days.filter((d) => d.hours.ours < d.hours[k]).length;
  const saving = 100 * (1 - mean("ours") / mean("driver"));
  const moved = days.map((d) => d.moved ?? 0);
  const kept = moved.filter((m) => m === 0).length;
  return (
    <div className="cards">
      <article className="card">
        <p className="big">−{saving.toFixed(1)} %</p>
        <p className="card-title">fleet hours vs real drivers</p>
        <p className="card-text">
          {hours(mean("driver"))} → {hours(mean("ours"))} of driving per day with the same vans, fewer on{" "}
          {oursBetter("driver")} of {days.length} days.
        </p>
      </article>
      <article className="card">
        <p className="big">
          {oursBetter("pyvrp")} / {days.length}
        </p>
        <p className="card-title">days better than PyVRP given 60 s</p>
        <p className="card-text">
          Fewer hours on average ({hours(mean("ours"))} vs {hours(mean("pyvrp"))}) in about two seconds, but only on
          half the days: PyVRP wins the smaller ones, ours the larger ones it cannot finish in a minute.
        </p>
      </article>
      <article className="card">
        <p className="big">{pct(100 * median(moved), 1)}</p>
        <p className="card-title">of stops change van (median day)</p>
        <p className="card-text">
          The vans' areas are far apart: on {kept} of {days.length} days our plan keeps exactly the drivers' groups.
          The saving comes mostly from the order of the stops inside each van.
        </p>
      </article>
    </div>
  );
}

interface PanelProps {
  doc: DayDoc;
  solver: string;
  other: string;
  onSolver: (key: string) => void;
  time: number;
  dark: boolean;
  onMap: (map: MLMap | null) => void;
}

function DayPanel({ doc, solver, other, onSolver, time, dark, onMap }: PanelProps) {
  const sol = doc.solvers[solver];
  const delta = sol.travel_h / doc.solvers[other].travel_h - 1;
  // the same area keeps its colour in both maps: vans are coloured after the drivers' vans
  const colors = useMemo(
    () => alignedColors(doc.solvers.driver.routes, sol.routes, VAN_COLORS),
    [doc, sol],
  );
  return (
    <section className="panel" aria-label={label(solver)}>
      <label className="panel-head">
        <span className="van-dots" aria-hidden="true">
          {colors.map((c, i) => (
            <span key={i} className="dot" style={{ background: c }} />
          ))}
        </span>
        <select value={solver} onChange={(e) => onSolver(e.target.value)}>
          {Object.keys(doc.solvers).map((k) => (
            <option key={k} value={k}>
              {label(k)}
            </option>
          ))}
        </select>
      </label>
      <RouteMap
        depot={doc.depot}
        stops={doc.stops}
        solution={sol}
        color={VAN_COLORS[0]}
        colors={colors}
        time={time}
        dark={dark}
        onMap={onMap}
      />
      <dl className="stats">
        <div>
          <dt>Fleet driving</dt>
          <dd>{hours(sol.travel_h)}</dd>
        </div>
        <div>
          <dt>Difference</dt>
          <dd className={delta < -0.001 ? "better" : delta > 0.001 ? "worse" : ""}>
            {Math.abs(delta) < 0.001
              ? "same"
              : `${delta < 0 ? "−" : "+"}${pct(Math.abs(100 * delta))} vs ${DAY_SOLVERS[other].name.toLowerCase()}`}
          </dd>
        </div>
        <div>
          <dt>Vans · late stops</dt>
          <dd>
            {sol.vans} · {sol.late}
          </dd>
        </div>
        <div>
          <dt>Compute</dt>
          <dd>{seconds(sol.runtime_s)}</dd>
        </div>
      </dl>
    </section>
  );
}

interface Props {
  days: Listing[];
  dayId: string | undefined;
  onDay: (id: string) => void;
  doc: DayDoc | null;
  dark: boolean;
  replay?: string; // id of the route or day to replay from the start once loaded (the chat's tour)
  onReplayed?: () => void;
}

export function StationDay({ days, dayId, onDay, doc, dark, replay, onReplayed }: Props) {
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

  if (days.length === 0) return null;
  return (
    <div id="day">
      <div className="controls">
        <RoutePicker routes={days} value={dayId} onChange={onDay} />
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
          <DayPanel doc={doc} solver={left} other={right} onSolver={setLeft} time={time} dark={dark} onMap={setMapA} />
          <DayPanel doc={doc} solver={right} other={left} onSolver={setRight} time={time} dark={dark} onMap={setMapB} />
        </div>
      )}
      <p className="note">
        Amazon publishes travel times only between stops of the same original route, so times between stops of
        different routes are estimated from distance (194 s + 99 s per km, fitted on the published ones); every plan is
        scored with the same matrix. Days with 3 to 6 vans and up to 1,200 stops. Our model runs on a GTX 1660 Super
        GPU, PyVRP on one CPU core for 60 s.
      </p>
    </div>
  );
}
