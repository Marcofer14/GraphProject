"""Benchmark on the standard synthetic distribution the neural models are trained on.

Uniform points, integer demands 1-9 and the Kool et al. capacities, so neural and classic
solvers are compared on exactly the problem the network learned. Classic solvers run in a
process pool; neural solvers run in the main process on the GPU, one instance at a time
(the latency a real-time re-planner would see).

    uv run nr-bench-synth --problem tsp --size 20 --n 200 \\
        --configs nearest_neighbor ortools:1 pyvrp:2 attention_greedy attention
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from neural_routing.core import VRPInstance
from neural_routing.data.synthetic import random_instance
from neural_routing.eval.benchmark import add_gaps, parse_config, run_instance, summarize
from neural_routing.ml.env import default_capacity
from neural_routing.ml.generators import generate_amazon_like
from neural_routing.solvers import get_solver

NEURAL = {"attention", "attention_greedy", "attention_ls"}
DEFAULT_CONFIGS = ["nearest_neighbor", "savings", "ortools:1", "pyvrp:2", "attention"]


def build(
    problem: str,
    size: int,
    seed: int,
    distribution: str = "uniform",
    time_windows: bool = False,
    tw_tightness: float = 1.0,
) -> VRPInstance:
    if distribution == "amazon":
        return build_amazon_like(problem, size, seed, time_windows, tw_tightness)
    if time_windows:
        raise ValueError("time windows are only generated for --distribution amazon")
    cap = math.inf if problem == "tsp" else default_capacity(size)
    inst = random_instance(size, seed=seed, capacity=cap, service_time=0.0)
    return VRPInstance(
        **{**inst.__dict__, "name": f"{problem}{size}_{seed}",
           "num_vehicles": 1 if problem == "tsp" else math.ceil(inst.demands.sum() / cap)}
    )  # fmt: skip


def build_amazon_like(
    problem: str, size: int, seed: int, time_windows: bool = False, tw_tightness: float = 1.0
) -> VRPInstance:
    """One calibrated instance (``ml.generators``) as a planar Euclidean VRPInstance.

    Coordinates use the depot-inclusive scaling, but any rescaling would do: every solver either
    works on the matrix or (neural) renormalises the coordinates the way its checkpoint expects,
    and gaps are relative. Time is in the same unit as distance (speed 1), scaled with it.
    """
    batch, times = generate_amazon_like(
        1, size, problem, generator=torch.Generator().manual_seed(seed), tw_tightness=tw_tightness
    )
    scale = 1000.0  # ~integer resolution for OR-Tools/PyVRP
    xy = batch.locs[0].double().numpy() * scale
    n1 = size + 1
    demand = batch.demand[0].double().numpy()
    cap = math.inf if problem == "tsp" else 1.0
    service, tw = np.zeros(n1), np.tile([0.0, np.inf], (n1, 1))
    if time_windows:
        service = times.service[0].double().numpy() * scale
        tw = times.tw[0].double().numpy() * scale
    return VRPInstance(
        name=f"amazon_{problem}{size}_{seed}",
        coords=xy,
        durations=np.linalg.norm(xy[:, None] - xy[None], axis=-1),
        demands=demand,
        service_times=service,
        time_windows=tw,
        capacity=cap,
        num_vehicles=1 if problem == "tsp" else max(1, math.ceil(demand.sum())),
    )


def _job(
    problem: str, size: int, seed: int, configs: list[str], *build_args: Any
) -> list[dict[str, Any]]:
    return run_instance(build(problem, size, seed, *build_args), configs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--problem", choices=["tsp", "cvrp"], default="tsp")
    parser.add_argument("--distribution", choices=["uniform", "amazon"], default="uniform")
    parser.add_argument("--time-windows", action="store_true", help="amazon only: keep windows")
    parser.add_argument("--tw-tightness", type=float, default=1.0, help="scale window widths")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--seed", type=int, default=10_000, help="first instance seed (not used in training)"
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--tag", default="", help="suffix for the output files")
    parser.add_argument(
        "--checkpoint", default=None, help="neural checkpoint (default: nearest size)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    classic = [c for c in args.configs if parse_config(c)[0] not in NEURAL]
    neural = [c for c in args.configs if parse_config(c)[0] in NEURAL]
    # per-config kwargs: "attention@path/to.pt" pins a checkpoint, else --checkpoint applies
    kwargs: dict[str, dict[str, Any]] = {}
    for cfg in neural:
        ckpt = cfg.split("@", 1)[1].split(":")[0] if "@" in cfg else args.checkpoint
        if ckpt:
            kwargs[cfg] = {"checkpoint": ckpt}
    dist = args.distribution
    build_args = (dist, args.time_windows, args.tw_tightness)
    seeds = range(args.seed, args.seed + args.n)
    rows: list[dict[str, Any]] = []

    if neural:  # warm up CUDA and load the checkpoints outside the timed region
        warm = build(args.problem, args.size, 0, *build_args)
        for cfg in neural:
            get_solver(parse_config(cfg)[0], **kwargs.get(cfg, {})).solve(warm)
        for s in tqdm(seeds, desc="neural"):
            rows += run_instance(build(args.problem, args.size, s, *build_args), neural, kwargs)

    if classic:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [
                pool.submit(_job, args.problem, args.size, s, classic, *build_args) for s in seeds
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="classic"):
                rows += f.result()

    df = add_gaps(pd.DataFrame(rows))
    summary = summarize(df)
    if not args.time_windows:
        summary = summary.drop(columns=["late_stops_%"])
    prefix = "synthetic" if dist == "uniform" else f"synthetic_{dist}"
    if args.time_windows:
        prefix += f"_tw{args.tw_tightness:g}"
    tag = f"{prefix}_{args.problem}{args.size}{'_' + args.tag if args.tag else ''}"
    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "raw" / f"{tag}.parquet", index=False)
    md = args.out / f"{tag}.md"
    md.write_text(
        f"# Synthetic benchmark · {args.problem.upper()}-{args.size} · {args.n} instances\n\n"
        f"{dist.capitalize()} instances, seeds not seen in training. "
        "Gap vs best solution found per instance.\n\n"
        + summary.to_markdown() + "\n",
        encoding="utf-8",
    )  # fmt: skip
    print(summary.to_string())
    print(f"\nwritten {md}")


if __name__ == "__main__":
    main()
