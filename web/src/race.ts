import { SOLVERS } from "./data";
import type { RouteDoc, SolutionData } from "./types";

export type Hardware = "gpu" | "cpu";

/** What a solver returns after `t` seconds of computing (one run per time budget). */
export interface Series {
  key: string;
  name: string;
  color: string;
  anytime: boolean; // classic solvers keep improving with more time; ours runs once
  points: { t: number; sol: SolutionData }[];
}

export const NO_ROUTE: SolutionData = { routes: [], times: [], travel_h: 0, controllable_h: 0, late: 0, runtime_s: 0 };

export function raceSeries(doc: RouteDoc, hardware: Hardware): Series[] {
  const once = (key: string, cpuKey: string) => {
    const sol = hardware === "gpu" ? doc.solvers[key] : doc.race[cpuKey][0];
    return [{ t: sol.runtime_s, sol }];
  };
  const anytime = (key: string) =>
    [...doc.race[key]].sort((a, b) => a.runtime_s - b.runtime_s).map((sol) => ({ t: sol.runtime_s, sol }));
  const series = (key: string, isAnytime: boolean, points: Series["points"]): Series => ({
    key,
    name: SOLVERS[key].name,
    color: SOLVERS[key].color,
    anytime: isAnytime,
    points,
  });
  return [
    series("ours", false, once("ours", "ours_cpu")),
    series("network", false, once("network", "network_cpu")),
    series("pyvrp", true, anytime("pyvrp")),
    series("ortools", true, anytime("ortools")),
  ];
}

/** The longest run that fits in the budget, or null if the solver has nothing yet. */
export function resultAt(s: Series, budget: number): { t: number; sol: SolutionData } | null {
  let found = null;
  for (const p of s.points) if (p.t <= budget) found = p;
  return found;
}

/** Shortest driving time any solver found on this route (GPU and CPU runs alike). */
export function bestTravel(doc: RouteDoc): number {
  const all = [
    ...Object.entries(doc.solvers).filter(([k]) => k !== "driver").map(([, s]) => s.travel_h),
    ...Object.values(doc.race).flat().map((p) => p.travel_h),
  ];
  return Math.min(...all);
}

const LOG_MIN = Math.log10(0.1);
const LOG_MAX = Math.log10(12);
/** Slider position 0..1000 <-> seconds on a log scale from 0.1 s to 12 s. */
export const toSeconds = (pos: number) => 10 ** (LOG_MIN + (pos / 1000) * (LOG_MAX - LOG_MIN));
export const toPosition = (s: number) => Math.round((1000 * (Math.log10(s) - LOG_MIN)) / (LOG_MAX - LOG_MIN));
