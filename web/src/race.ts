import { SOLVERS } from "./data";
import type { RouteDoc, SolutionData } from "./types";

export type Hardware = "gpu" | "cpu";

/** One run: the time it was given (`budget`, or its own time for a one-shot answer), the time it
 * actually took (`t`, a few ms over the budget) and what it returned. */
export interface RacePointView {
  budget: number;
  t: number;
  sol: SolutionData;
}

/** What a solver returns for each time budget (one run per budget). */
export interface Series {
  key: string;
  name: string;
  color: string;
  anytime: boolean; // keeps improving with more time; the network alone answers once
  points: RacePointView[];
}

export const NO_ROUTE: SolutionData = { routes: [], times: [], travel_h: 0, controllable_h: 0, late: 0, runtime_s: 0 };

export function raceSeries(doc: RouteDoc, hardware: Hardware): Series[] {
  const once = (key: string, cpuKey: string) => {
    const sol = hardware === "gpu" ? doc.solvers[key] : doc.race[cpuKey][0];
    return [{ budget: sol.runtime_s, t: sol.runtime_s, sol }];
  };
  const anytime = (key: string) =>
    [...doc.race[key]]
      .sort((a, b) => a.runtime_s - b.runtime_s)
      .map((sol) => ({ budget: sol.budget_s ?? sol.runtime_s, t: sol.runtime_s, sol }));
  const series = (key: string, isAnytime: boolean, points: Series["points"]): Series => ({
    key,
    name: SOLVERS[key].name,
    color: SOLVERS[key].color,
    anytime: isAnytime,
    points,
  });
  // with the ILS export our model is anytime too: first answer, then the search until the budget
  const ilsKey = hardware === "gpu" ? "ours_ils" : "ours_ils_cpu";
  const ours =
    doc.race[ilsKey]?.length > 1
      ? series("ours", true, anytime(ilsKey))
      : series("ours", false, once("ours", "ours_cpu"));
  return [
    ours,
    series("network", false, once("network", "network_cpu")),
    series("pyvrp", true, anytime("pyvrp")),
    series("ortools", true, anytime("ortools")),
  ];
}

/** The run with the largest budget that fits, or null if the solver has nothing yet. */
export function resultAt(s: Series, budget: number): RacePointView | null {
  let found = null;
  for (const p of s.points) if (p.budget <= budget) found = p;
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
