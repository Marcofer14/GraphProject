import type { DemoIndex, RouteDoc } from "./types";

const base = `${import.meta.env.BASE_URL}data`;

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${base}/${path}`);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return (await res.json()) as T;
}

export const loadIndex = () => getJson<DemoIndex>("index.json");
export const loadRoute = (id: string) => getJson<RouteDoc>(`routes/${id}.json`);

// Display order, names and colours, shared by the table, the maps and the legend.
export const SOLVERS: Record<string, { name: string; color: string; note?: string }> = {
  ours: { name: "Our model", color: "#534AB7", note: "network + local search" },
  network: { name: "Network alone", color: "#7F77DD", note: "no local search" },
  pyvrp_fast: { name: "PyVRP", color: "#D85A30", note: "0.5 s" },
  pyvrp_1s: { name: "PyVRP", color: "#D85A30", note: "1 s" },
  pyvrp: { name: "PyVRP", color: "#D85A30", note: "10 s" },
  ortools_fast: { name: "OR-Tools", color: "#1D9E75", note: "0.5 s" },
  ortools: { name: "OR-Tools", color: "#1D9E75", note: "10 s" },
  driver: { name: "Real driver", color: "#BA7517" },
};

export const ROUTE_CHOICES = ["driver", "ours", "network", "ortools", "pyvrp"];
