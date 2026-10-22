"""Export benchmark results as static JSON for the web demo (``web/public/data``).

Nothing is computed when someone opens the demo: every route, metric and "race" point is
written here once. Three kinds of files:

* ``index.json`` - headline numbers and the list of demo routes and station-days.
* ``routes/<id>.json`` - one real route: stops, what each solver did (sequence, arrival times,
  metrics) and the race (best route each solver returns given 0.5 ... 10 s of computing).
* ``days/<id>.json`` - one whole station-day re-planned by the fleet solvers.

Demo routes are drawn at random per city from the phase-4 evaluation (not cherry-picked)::

    uv run nr-demo-export --results results/raw/route_evaluation_n200_phase4_*.parquet \\
        --days results/raw/station-day_training_n6_fleet_*.parquet \\
        --equal-time results/raw/route_evaluation_n200_equal_time_*.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_routing.core import Solution, VRPInstance, evaluate
from neural_routing.data.amazon import AmazonDataset
from neural_routing.eval.benchmark import add_gaps, unavoidable_s
from neural_routing.solvers import get_solver

log = logging.getLogger(__name__)
Json = dict[str, Any]

CITIES = {"DLA": "Los Angeles", "DSE": "Seattle", "DBO": "Boston", "DAU": "Austin",
          "DCH": "Chicago"}  # fmt: skip
CHECKPOINT = "checkpoints/cvrp100_pomo_real.pt"

# benchmark config -> (demo key, label shown on the page)
ROUTE_SOLVERS = {
    "reference": ("driver", "Real driver"),
    f"attention_ls@{CHECKPOINT}": ("ours", "Our model"),
    f"attention@{CHECKPOINT}": ("network", "Network alone"),
    "ortools:10": ("ortools", "OR-Tools"),
    "pyvrp:10": ("pyvrp", "PyVRP"),
}
EQUAL_TIME_SOLVERS = {  # same budget as our model (nr-bench --tag equal_time)
    "ortools:0.5": ("ortools_fast", "OR-Tools (0.5 s)"),
    "pyvrp:0.5": ("pyvrp_fast", "PyVRP (0.5 s)"),
    "pyvrp:1": ("pyvrp_1s", "PyVRP (1 s)"),
}
DAY_SOLVERS = {
    "reference": ("driver", "Real drivers"),
    f"cluster_attention_ls@{CHECKPOINT}": ("ours", "Our model (cluster + network)"),
    "pyvrp:60": ("pyvrp", "PyVRP (60 s)"),
    "savings": ("savings", "Savings heuristic"),
}


def city(station: str) -> str:
    return CITIES.get(station[:3], station)


def schedule(instance: VRPInstance, route: list[int]) -> list[float]:
    """Seconds after departure at which each stop is reached, then the return to the depot
    (same clock as :func:`core.evaluate`: wait if early, then serve)."""
    D, tw = instance.durations, instance.time_windows
    t, prev, out = 0.0, 0, []
    for node in route:
        t = max(t + D[prev, node], tw[node, 0])
        out.append(round(t))
        t += instance.service_times[node]
        prev = node
    out.append(round(t + D[prev, 0]))
    return out


def solution_payload(instance: VRPInstance, routes: list[list[int]], runtime_s: float) -> Json:
    m = evaluate(instance, Solution.from_lists(routes))
    ctrl = m.travel_time_s - unavoidable_s(instance)
    return {
        "routes": routes,
        "times": [schedule(instance, r) for r in routes],
        "travel_h": round(m.travel_time_s / 3600, 3),
        "controllable_h": round(ctrl / 3600, 3),
        "late": m.tw_violations,
        "runtime_s": round(runtime_s, 3),
    }


def pick_routes(df: pd.DataFrame, per_city: int, seed: int) -> list[str]:
    """``per_city`` random routes per city, so the demo is not a hand-picked best case."""
    rng = np.random.default_rng(seed)
    routes = df.drop_duplicates("instance")[["instance", "station"]]
    picked: list[str] = []
    for _, g in routes.groupby(routes["station"].map(city), sort=True):
        ids = sorted(g["instance"])
        picked += list(rng.choice(ids, size=min(per_city, len(ids)), replace=False))
    return [str(i) for i in picked]


def _classic_race(split: str, route_id: str, limits: list[float]) -> dict[str, list[Json]]:
    inst = AmazonDataset(split).route_instance(route_id)
    race: dict[str, list[Json]] = {}
    for solver in ("ortools", "pyvrp"):
        race[solver] = []
        for limit in limits:
            sol = get_solver(solver).solve(inst, time_limit=limit)
            race[solver].append({"budget_s": limit} | solution_payload(
                inst, [list(r) for r in sol.routes], sol.runtime_s))  # fmt: skip
    return race


def _neural_cpu(instance: VRPInstance) -> dict[str, Json]:
    """The network and the hybrid on CPU, to show the speed is not only a GPU effect."""
    out = {}
    for key, name in (("network_cpu", "attention"), ("ours_cpu", "attention_ls")):
        solver = get_solver(name, checkpoint=CHECKPOINT, device="cpu")
        t0 = time.perf_counter()
        sol = solver.solve(instance)
        out[key] = solution_payload(instance, [list(r) for r in sol.routes],
                                    time.perf_counter() - t0)  # fmt: skip
    return out


def export_routes(
    df: pd.DataFrame, split: str, ids: list[str], limits: list[float], out: Path, workers: int
) -> list[Json]:
    ds = AmazonDataset(split)
    instances = {rid: ds.route_instance(rid) for rid in ids}
    get_solver("attention", checkpoint=CHECKPOINT, device="cpu").solve(instances[ids[0]])  # warm
    cpu = {rid: _neural_cpu(inst) for rid, inst in instances.items()}  # before the pool starts
    with ProcessPoolExecutor(workers) as pool:
        races = dict(zip(ids, pool.map(_classic_race, [split] * len(ids), ids,
                                       [limits] * len(ids)), strict=True))  # fmt: skip

    (out / "routes").mkdir(parents=True, exist_ok=True)
    listing = []
    for rid in ids:
        inst = instances[rid]
        rows = df[df["instance"] == rid].set_index("config")
        solvers = {}
        for cfg, (key, label) in ROUTE_SOLVERS.items():
            row = rows.loc[cfg].to_dict()
            payload = solution_payload(inst, json.loads(row["routes"]), float(row["runtime_s"]))
            score = row["driver_score"]
            solvers[key] = {"label": label, "driver_score": None if pd.isna(score)
                            else round(float(score), 3)} | payload  # fmt: skip
        race = races[rid]
        for key in ("ortools", "pyvrp"):  # the 10 s point is the benchmark run itself
            race[key].append(
                {"budget_s": 10.0}
                | {k: v for k, v in solvers[key].items() if k not in ("label", "driver_score")}
            )
        race |= {key: [{"budget_s": None} | value] for key, value in cpu[rid].items()}
        station = str(inst.meta["station"])
        doc = {
            "id": rid,
            "city": city(station),
            "station": station,
            "date": str(inst.meta["date"]),
            "depot": [round(float(v), 5) for v in inst.coords[0]],
            "stops": [[round(float(a), 5), round(float(b), 5)] for a, b in inst.coords[1:]],
            "zones": [str(z) for z in inst.meta["zones"][1:]],
            "fixed_h": round(unavoidable_s(inst) / 3600, 3),
            "solvers": solvers,
            "race": race,
        }
        (out / "routes" / f"{rid}.json").write_text(json.dumps(doc, separators=(",", ":")))
        listing.append({
            "id": rid, "city": doc["city"], "stops": inst.num_customers,
            "hours": {k: v["travel_h"] for k, v in solvers.items()},
        })  # fmt: skip
        log.info("route %s (%s, %d stops) written", rid, doc["city"], inst.num_customers)
    return listing


def export_days(days: pd.DataFrame, split: str, out: Path) -> list[Json]:
    ds = AmazonDataset(split)
    model = ds.fit_travel_model(seed=0)  # same model as the benchmark (nr-bench --seed 0)
    (out / "days").mkdir(parents=True, exist_ok=True)
    listing = []
    for name, rows in days.groupby("instance", sort=True):
        station, date = str(name).split("_", 1)
        inst = ds.station_day_instance(station, date, model)
        rows = rows.set_index("config")
        solvers = {}
        for cfg, (key, label) in DAY_SOLVERS.items():
            if cfg not in rows.index:
                continue
            row = rows.loc[cfg].to_dict()
            routes = json.loads(row["routes"])
            solvers[key] = {"label": label, "vans": len(routes)} | solution_payload(
                inst, routes, float(row["runtime_s"]))  # fmt: skip
        doc = {
            "id": str(name), "city": city(station), "station": station, "date": date,
            "depot": [round(float(v), 5) for v in inst.coords[0]],
            "stops": [[round(float(a), 5), round(float(b), 5)] for a, b in inst.coords[1:]],
            "solvers": solvers,
        }  # fmt: skip
        (out / "days" / f"{name}.json").write_text(json.dumps(doc, separators=(",", ":")))
        listing.append({"id": str(name), "city": doc["city"], "stops": inst.num_customers,
                        "hours": {k: v["travel_h"] for k, v in solvers.items()}})  # fmt: skip
        log.info("station-day %s (%d stops) written", name, inst.num_customers)
    return listing


def headline(df: pd.DataFrame) -> dict[str, Any]:
    """Means over all evaluation routes (not just the demo ones), and on how many routes our
    model is strictly better than each other solver (by more than half a second)."""
    travel = df.pivot_table(index="instance", columns="config", values="travel_time_s")
    ours = travel[next(c for c, (k, _) in ROUTE_SOLVERS.items() if k == "ours")]
    out = {}
    for cfg, (key, label) in (ROUTE_SOLVERS | EQUAL_TIME_SOLVERS).items():
        if cfg not in travel:
            continue
        g = df[df["config"] == cfg]
        out[key] = {
            "label": label,
            "travel_h": round(float(g["travel_time_s"].mean()) / 3600, 3),
            "gap_ctrl_pct": round(float(g["gap_ctrl_pct"].mean()), 2),
            "late_pct": round(100 * float(g["tw_violations"].sum() / g["num_customers"].sum()), 2),
            "runtime_s": round(float(g["runtime_s"].median()), 3),
        }
        if key != "ours":
            out[key]["ours_wins"] = int((ours < travel[cfg] - 0.5).sum())
    return {"routes": int(df["instance"].nunique()), "solvers": out}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--results", type=Path, required=True, help="phase-4 route parquet")
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--days", type=Path, default=None, help="station-day fleet parquet")
    parser.add_argument("--days-split", default="training")
    parser.add_argument("--per-city", type=int, default=5)
    parser.add_argument("--limits", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--equal-time", type=Path, default=None, help="equal-time route parquet")
    parser.add_argument("--index-only", action="store_true",
                        help="only rewrite the headline in an existing index.json")  # fmt: skip
    parser.add_argument("--out", type=Path, default=Path("web/public/data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(args.results)
    if args.equal_time:  # same routes, more configs: gaps against the best of all of them
        extra = pd.read_parquet(args.equal_time)
        df = add_gaps(pd.concat([df, extra]).drop(columns=["gap_pct", "gap_ctrl_pct"]))
    if args.index_only:
        path = args.out / "index.json"
        index = json.loads(path.read_text()) | {"headline": headline(df)}
        path.write_text(json.dumps(index, indent=1))
        log.info("headline rewritten in %s", path)
        return
    ids = pick_routes(df, args.per_city, args.seed)
    log.info("exporting %d routes: %s", len(ids), ids)
    args.out.mkdir(parents=True, exist_ok=True)
    index = {
        "generated": time.strftime("%Y-%m-%d"),
        "headline": headline(df),
        "routes": export_routes(df, args.split, ids, args.limits, args.out, args.workers),
        "days": export_days(pd.read_parquet(args.days), args.days_split, args.out)
        if args.days else [],
    }  # fmt: skip
    (args.out / "index.json").write_text(json.dumps(index, indent=1))
    log.info("written %s", args.out / "index.json")


if __name__ == "__main__":
    main()
