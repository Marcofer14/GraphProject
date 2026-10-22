import type { LatLng, SolutionData } from "./types";

export type LngLat = [number, number];

export const toLngLat = ([lat, lng]: LatLng): LngLat => [lng, lat];

/** One van's path: depot, its stops in order, depot; with the time each point is reached. */
export interface Track {
  points: LngLat[];
  times: number[];
}

export function tracks(depot: LatLng, stops: LatLng[], sol: SolutionData): Track[] {
  const home = toLngLat(depot);
  return sol.routes.map((route, v) => ({
    points: [home, ...route.map((node) => toLngLat(stops[node - 1])), home],
    times: [0, ...sol.times[v]],
  }));
}

export const endTime = (ts: Track[]) => Math.max(0, ...ts.map((t) => t.times[t.times.length - 1]));

/** The part of a track driven by time `t`, ending at the van's interpolated position.
 * Service time at a stop is folded into the leg that follows it. */
export function drivenBy(track: Track, t: number): LngLat[] {
  const { points, times } = track;
  const i = times.findIndex((x) => x > t);
  if (i === -1) return points;
  if (i === 0) return [points[0]];
  const f = (t - times[i - 1]) / (times[i] - times[i - 1]);
  const [a, b] = [points[i - 1], points[i]];
  return [...points.slice(0, i), [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]];
}

/** Arrival time per stop (index = node - 1), for colouring visited stops. */
export function arrivals(numStops: number, sol: SolutionData): number[] {
  const out = new Array<number>(numStops).fill(Infinity);
  sol.routes.forEach((route, v) => route.forEach((node, k) => (out[node - 1] = sol.times[v][k])));
  return out;
}

/** Which van serves each stop (index = node - 1), -1 if none. */
export function vanOf(numStops: number, sol: SolutionData): number[] {
  const out = new Array<number>(numStops).fill(-1);
  sol.routes.forEach((route, v) => route.forEach((node) => (out[node - 1] = v)));
  return out;
}

/** Colours for `routes` so that each van takes the colour of the reference van it shares most
 * stops with (the same area keeps its colour across plans); leftovers take unused colours. */
export function alignedColors(reference: number[][], routes: number[][], palette: string[]): string[] {
  const owner = new Map<number, number>();
  reference.forEach((route, v) => route.forEach((stop) => owner.set(stop, v)));
  const used = new Set<number>();
  const pick = routes.map((route) => {
    const counts = new Map<number, number>();
    route.forEach((stop) => {
      const v = owner.get(stop);
      if (v !== undefined) counts.set(v, (counts.get(v) ?? 0) + 1);
    });
    const best = [...counts.entries()].sort((a, b) => b[1] - a[1]).find(([v]) => !used.has(v))?.[0];
    if (best !== undefined) used.add(best);
    return best;
  });
  let next = 0;
  return pick.map((v) => {
    if (v !== undefined) return palette[v % palette.length];
    while (used.has(next)) next++;
    used.add(next);
    return palette[next % palette.length];
  });
}
