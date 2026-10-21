"""Benchmark solvers on Amazon instances and write a results table.

Examples::

    # 200 single-van routes: driver vs heuristics vs OR-Tools vs PyVRP
    uv run nr-bench --mode route --n 200 --configs reference nearest_neighbor savings \\
        ortools:1 ortools:10 pyvrp:1 pyvrp:10

    # whole station-days re-assigned from scratch
    uv run nr-bench --mode station-day --n 5 --configs reference savings ortools:60 pyvrp:60

A config is ``solver`` or ``solver:time_limit_seconds``. Gap is measured against the best
complete solution found for each instance by any config in the run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from neural_routing.core import VRPInstance, evaluate
from neural_routing.data.amazon import AmazonDataset, TravelTimeModel
from neural_routing.solvers import get_solver

log = logging.getLogger(__name__)


def parse_config(cfg: str) -> tuple[str, float]:
    name, _, limit = cfg.partition(":")
    return name, float(limit) if limit else 1.0


def run_instance(
    instance: VRPInstance,
    configs: Iterable[str],
    solver_kwargs: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for cfg in configs:
        name, limit = parse_config(cfg)
        row: dict[str, Any] = {
            "instance": instance.name,
            "config": cfg,
            "solver": name,
            "time_limit_s": limit,
            "num_customers": instance.num_customers,
            "num_vehicles_real": instance.num_vehicles,
            "route_score": instance.meta.get("route_score"),
            "station": instance.meta.get("station"),
        }
        try:
            solver = get_solver(name, **(solver_kwargs or {}).get(name, {}))
            sol = solver.solve(instance, time_limit=limit)
            row |= evaluate(instance, sol).as_dict()
        except Exception as exc:  # a failing solver must not kill a long benchmark
            log.warning("%s on %s failed: %s", cfg, instance.name, exc)
            row["error"] = str(exc)
        rows.append(row)
    return rows


def _route_job(split: str, route_id: str, configs: list[str]) -> list[dict[str, Any]]:
    return run_instance(AmazonDataset(split).route_instance(route_id), configs)


def _station_day_job(
    split: str, station: str, date: str, model: TravelTimeModel, configs: list[str]
) -> list[dict[str, Any]]:
    return run_instance(AmazonDataset(split).station_day_instance(station, date, model), configs)


def add_gaps(df: pd.DataFrame) -> pd.DataFrame:
    ok = df["is_complete"].fillna(False).astype(bool)
    best = df[ok].groupby("instance")["travel_time_s"].min()
    df["gap_pct"] = 100 * (df["travel_time_s"] / df["instance"].map(best) - 1)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("config", sort=False)
    out = pd.DataFrame(
        {
            "instances": g["instance"].nunique(),
            "travel_time_h": g["travel_time_s"].mean() / 3600,
            "gap_%": g["gap_pct"].mean(),
            "gap_%_p90": g["gap_pct"].quantile(0.9),
            "late_stops_%": 100 * g["tw_violations"].sum() / g["num_customers"].sum(),
            "vehicles": g["num_routes"].mean(),
            "runtime_s_p50": g["runtime_s"].median(),
            "failures": g["is_complete"].apply(
                lambda s: int((~s.fillna(False).astype(bool)).sum())
            ),
        }
    )
    return out.sort_values("gap_%").round(2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--split", default="training")
    parser.add_argument("--mode", choices=["route", "station-day"], default="route")
    parser.add_argument("--n", type=int, default=50, help="number of instances")
    parser.add_argument("--configs", nargs="+", default=["reference", "nearest_neighbor", "savings",
                                                         "ortools:1", "pyvrp:1"])  # fmt: skip
    parser.add_argument("--route-score", default=None, help="only routes with this driver score")
    parser.add_argument("--max-stops", type=int, default=None, help="station-day size limit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ds = AmazonDataset(args.split)
    rng = np.random.default_rng(args.seed)
    with ProcessPoolExecutor(args.workers) as pool:
        if args.mode == "route":
            routes = ds.routes
            if args.route_score:
                routes = routes[routes["route_score"] == args.route_score]
            ids = rng.choice(routes.index.to_numpy(), size=min(args.n, len(routes)), replace=False)
            futures = [pool.submit(_route_job, args.split, rid, args.configs) for rid in ids]
        else:
            model = ds.fit_travel_model(seed=args.seed)
            log.info("travel model: %.1f s + %.1f s/km", model.intercept, model.slope)
            days = ds.station_days()
            if args.max_stops:
                days = days[days["num_stops"] <= args.max_stops]
            days = days.sample(n=min(args.n, len(days)), random_state=args.seed)
            futures = [
                pool.submit(_station_day_job, args.split, str(st), str(date), model, args.configs)
                for st, date in zip(days["station_code"], days["date"], strict=True)
            ]
        rows = [row for f in tqdm(as_completed(futures), total=len(futures)) for row in f.result()]

    df = add_gaps(pd.DataFrame(rows))
    summary = summarize(df)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    tag = f"{args.mode}_{args.split}_n{len(df['instance'].unique())}"
    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "raw" / f"{tag}_{stamp}.parquet", index=False)
    md = args.out / f"{tag}.md"
    md.write_text(
        f"# Benchmark `{args.mode}` · {args.split} · {df['instance'].nunique()} instances\n\n"
        f"Generated {stamp} with `nr-bench {' '.join(sys.argv[1:])}`\n\n"
        + summary.to_markdown()
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string())
    print(f"\nwritten {md}")


if __name__ == "__main__":
    main()
