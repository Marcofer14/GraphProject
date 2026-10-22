import type { MouseEvent } from "react";

import { type Series } from "../race";

const W = 680;
const H = 280;
const M = { left: 64, right: 92, top: 16, bottom: 40 };
const X_MIN = 0.1;
const X_MAX = 12;
const X_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10];

const xs = (t: number) =>
  M.left +
  ((Math.log10(Math.max(t, X_MIN)) - Math.log10(X_MIN)) / (Math.log10(X_MAX) - Math.log10(X_MIN))) *
    (W - M.left - M.right);

const LABEL_GAP = 17;

/** Push label positions apart so no two are closer than LABEL_GAP, keeping their order. */
function spread(labels: { key: string; y: number }[], bottom: number): Map<string, number> {
  const sorted = [...labels].sort((a, b) => a.y - b.y).map((l) => ({ ...l }));
  sorted.forEach((l, i) => {
    if (i > 0) l.y = Math.max(l.y, sorted[i - 1].y + LABEL_GAP);
  });
  const overflow = Math.max(0, (sorted.at(-1)?.y ?? 0) - bottom);
  return new Map(sorted.map((l) => [l.key, l.y - overflow]));
}

function niceStep(max: number) {
  return [1, 2, 5, 10, 15, 20, 30, 60].find((s) => max / s <= 5) ?? 60;
}

interface Props {
  series: Series[];
  best: number; // hours
  driverH: number;
  budget: number;
  onBudget: (s: number) => void;
}

export function RaceChart({ series, best, driverH, budget, onBudget }: Props) {
  const minutes = (h: number) => (h - best) * 60;
  const driverMin = minutes(driverH);
  const all = series.flatMap((s) => s.points.map((p) => minutes(p.sol.travel_h)));
  const step = niceStep(Math.max(driverMin, ...all) * 1.1);
  const yMax = step * Math.ceil((Math.max(driverMin, ...all) * 1.1) / step);
  const ys = (m: number) => M.top + (1 - m / yMax) * (H - M.top - M.bottom);
  const yTicks = Array.from({ length: Math.floor(yMax / step) + 1 }, (_, i) => i * step);
  const lastY = (s: Series) => ys(minutes(s.points[s.points.length - 1].sol.travel_h));
  // labels at the right edge (driver, anytime solvers) and next to the one-shot points
  const right = spread(
    [{ key: "driver", y: ys(driverMin) }, ...series.filter((s) => s.anytime).map((s) => ({ key: s.key, y: lastY(s) }))],
    H - M.bottom,
  );
  const inline = spread(
    series.filter((s) => !s.anytime).map((s) => ({ key: s.key, y: lastY(s) })),
    H - M.bottom,
  );

  const click = (e: MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - box.left) / box.width) * W;
    const f = (x - M.left) / (W - M.left - M.right);
    if (f < 0 || f > 1) return;
    onBudget(10 ** (Math.log10(X_MIN) + f * (Math.log10(X_MAX) - Math.log10(X_MIN))));
  };

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="Minutes of driving above the best route, against seconds of computing"
      onClick={click}
    >
      {yTicks.map((m) => (
        <g key={m}>
          <line x1={M.left} x2={W - M.right} y1={ys(m)} y2={ys(m)} className="grid" />
          <text x={M.left - 8} y={ys(m)} className="tick" textAnchor="end" dominantBaseline="central">
            {m === 0 ? "best" : `+${m} min`}
          </text>
        </g>
      ))}
      {X_TICKS.map((t) => (
        <text key={t} x={xs(t)} y={H - M.bottom + 18} className="tick" textAnchor="middle">
          {t} s
        </text>
      ))}
      <text x={(M.left + W - M.right) / 2} y={H - 6} className="axis" textAnchor="middle">
        seconds of computing (log scale)
      </text>

      <line x1={M.left} x2={W - M.right} y1={ys(driverMin)} y2={ys(driverMin)} className="driver" />
      <text x={W - M.right + 6} y={right.get("driver")} className="label" dominantBaseline="central">
        Real driver
      </text>

      {series.map((s) => {
        const pts = s.points.map((p) => [xs(p.t), ys(minutes(p.sol.travel_h))] as const);
        const last = pts[pts.length - 1];
        return (
          <g key={s.key} style={{ color: s.color }}>
            {s.anytime && pts.length > 1 && (
              <polyline points={pts.map((p) => p.join(",")).join(" ")} className="series" />
            )}
            {pts.map(([x, y], i) => (
              <circle key={i} cx={x} cy={y} r={s.anytime ? 4 : 6} className="point" />
            ))}
            <text
              x={s.anytime ? W - M.right + 6 : last[0] + 9}
              y={(s.anytime ? right : inline).get(s.key)}
              className="label"
              dominantBaseline="central"
            >
              {s.name}
            </text>
          </g>
        );
      })}

      <line x1={xs(budget)} x2={xs(budget)} y1={M.top} y2={H - M.bottom} className="budget" />
    </svg>
  );
}
