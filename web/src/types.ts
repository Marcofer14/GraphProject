// Shapes of the JSON written by `nr-demo-export` (src/neural_routing/eval/demo_export.py).

export type LatLng = [number, number];

export interface SolutionData {
  routes: number[][]; // customer node ids (1..N) per van; depot implicit at both ends
  times: number[][]; // seconds after departure at each stop, then the return to the depot
  travel_h: number;
  controllable_h: number;
  late: number;
  runtime_s: number;
}

export interface SolverResult extends SolutionData {
  label: string;
  driver_score?: number | null;
  vans?: number;
}

export interface RacePoint extends SolutionData {
  budget_s: number | null; // null: our model, which has no time budget
}

export interface RouteDoc {
  id: string;
  city: string;
  station: string;
  date: string;
  depot: LatLng;
  stops: LatLng[];
  zones: string[];
  fixed_h: number;
  solvers: Record<string, SolverResult>;
  race: Record<string, RacePoint[]>;
}

export interface HeadlineSolver {
  label: string;
  travel_h: number;
  gap_ctrl_pct: number;
  late_pct: number;
  runtime_s: number;
  ours_wins?: number; // routes where our model is strictly better
}

export interface Listing {
  id: string;
  city: string;
  stops: number;
  hours: Record<string, number>;
}

export interface DemoIndex {
  generated: string;
  headline: { routes: number; solvers: Record<string, HeadlineSolver> };
  routes: Listing[];
  days: Listing[];
}
