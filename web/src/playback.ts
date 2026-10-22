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
