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
import json
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
from neural_routing.core.driver_score import driver_score
from neural_routing.data.amazon import AmazonDataset, TravelTimeModel
from neural_routing.solvers import get_solver

log = logging.getLogger(__name__)

# solvers that run on the GPU in the main process (one model load, no per-worker CUDA context)
NEURAL = {"attention", "attention_greedy", "attention_ls", "attention_ils", "cluster_attention_ls"}


def parse_config(cfg: str) -> tuple[str, float]:
    """``solver[@checkpoint][:time_limit]`` -> (solver, time_limit)."""
    head, _, limit = cfg.rpartition(":") if ":" in cfg.split("@")[-1] else (cfg, "", "")
    return head.split("@", 1)[0], float(limit) if limit else 1.0


def neural_kwargs(configs: Iterable[str], default_checkpoint: str | None = None) -> dict[str, Any]:
    """Per-config solver kwargs: ``attention@path.pt`` pins a checkpoint."""
    out: dict[str, Any] = {}
    for cfg in configs:
        if parse_config(cfg)[0] not in NEURAL:
            continue
        ckpt = cfg.split("@", 1)[1].split(":")[0] if "@" in cfg else default_checkpoint
        if ckpt:
            out[cfg] = {"checkpoint": ckpt}
    return out


def unavoidable_s(instance: VRPInstance) -> float:
    """Lower bound shared by every solution: leave the depot and come back at least once."""
    D = instance.durations
    return float(D[0, 1:].min() + D[1:, 0].min())


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
            "fixed_s": unavoidable_s(instance),
        }
        try:
            kw = solver_kwargs or {}
            solver = get_solver(name, **kw.get(cfg, kw.get(name, {})))
            sol = solver.solve(instance, time_limit=limit)
            row |= evaluate(instance, sol).as_dict()
            row["driver_score"] = driver_score(instance, sol)
            row["routes"] = json.dumps([list(r) for r in sol.routes])  # recompute metrics later
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
    """Gap vs the best complete solution per instance, on the full tour and on its controllable
    part (tour minus the unavoidable depot round trip, which dominates last-mile routes)."""
    ok = df["is_complete"].fillna(False).astype(bool)
    best = df[ok].groupby("instance")["travel_time_s"].min()
    df["gap_pct"] = 100 * (df["travel_time_s"] / df["instance"].map(best) - 1)
    if "fixed_s" in df:
        ctrl = df["travel_time_s"] - df["fixed_s"]
        best_ctrl = df["instance"].map(best) - df["fixed_s"]
        df["gap_ctrl_pct"] = 100 * (ctrl / best_ctrl - 1)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("config", sort=False)
    out = pd.DataFrame(
        {
            "instances": g["instance"].nunique(),
            "travel_time_h": g["travel_time_s"].mean() / 3600,
            "gap_%": g["gap_pct"].mean(),
            "gap_%_p90": g["gap_pct"].quantile(0.9),
            **({"gap_ctrl_%": g["gap_ctrl_pct"].mean()} if "gap_ctrl_pct" in df else {}),
            **(
                {"driver_score": g["driver_score"].mean()}
                if "driver_score" in df and df["driver_score"].notna().any()
                else {}
            ),
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
    parser.add_argument(
        "--min-routes", type=int, default=1, help="station-days with this many vans or more"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--tag", default="", help="suffix for the output files")
    parser.add_argument("--checkpoint", default=None, help="default neural checkpoint")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ds = AmazonDataset(args.split)
    rng = np.random.default_rng(args.seed)
    neural = [c for c in args.configs if parse_config(c)[0] in NEURAL]
    classic = [c for c in args.configs if c not in neural]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(args.workers) as pool:
        if args.mode == "route":
            routes = ds.routes
            if args.route_score:
                routes = routes[routes["route_score"] == args.route_score]
            ids = rng.choice(routes.index.to_numpy(), size=min(args.n, len(routes)), replace=False)
            futures = [pool.submit(_route_job, args.split, rid, classic) for rid in ids]
            if neural:  # GPU, main process, while the pool works on the classic solvers
                kwargs = neural_kwargs(neural, args.checkpoint)
                for rid in tqdm(ids, desc="neural"):
                    rows += run_instance(ds.route_instance(str(rid)), neural, kwargs)
        else:
            model = ds.fit_travel_model(seed=args.seed)
            log.info("travel model: %.1f s + %.1f s/km", model.intercept, model.slope)
            days = ds.station_days()
            if args.max_stops:
                days = days[days["num_stops"] <= args.max_stops]
            days = days[days["num_routes"] >= args.min_routes]
            days = days.sample(n=min(args.n, len(days)), random_state=args.seed)
            pairs = [
                (str(st), str(date))
                for st, date in zip(days["station_code"], days["date"], strict=True)
            ]
            futures = [
                pool.submit(_station_day_job, args.split, st, date, model, classic)
                for st, date in pairs
            ]
            if neural:  # GPU, main process, while the pool works on the classic solvers
                kwargs = neural_kwargs(neural, args.checkpoint)
                for st, date in tqdm(pairs, desc="neural"):
                    inst = ds.station_day_instance(st, date, model)
                    rows += run_instance(inst, neural, kwargs)
        rows += [row for f in tqdm(as_completed(futures), total=len(futures)) for row in f.result()]

    df = add_gaps(pd.DataFrame(rows))
    summary = summarize(df)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    tag = f"{args.mode}_{args.split}_n{len(df['instance'].unique())}"
    tag += f"_{args.tag}" if args.tag else ""
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
