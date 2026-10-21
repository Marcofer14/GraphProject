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

import pandas as pd
from tqdm import tqdm

from neural_routing.core import VRPInstance
from neural_routing.data.synthetic import random_instance
from neural_routing.eval.benchmark import add_gaps, parse_config, run_instance, summarize
from neural_routing.ml.env import default_capacity
from neural_routing.solvers import get_solver

NEURAL = {"attention", "attention_greedy"}
DEFAULT_CONFIGS = ["nearest_neighbor", "savings", "ortools:1", "pyvrp:2", "attention"]


def build(problem: str, size: int, seed: int) -> VRPInstance:
    cap = math.inf if problem == "tsp" else default_capacity(size)
    inst = random_instance(size, seed=seed, capacity=cap, service_time=0.0)
    return VRPInstance(
        **{**inst.__dict__, "name": f"{problem}{size}_{seed}",
           "num_vehicles": 1 if problem == "tsp" else math.ceil(inst.demands.sum() / cap)}
    )  # fmt: skip


def _job(problem: str, size: int, seed: int, configs: list[str]) -> list[dict[str, Any]]:
    return run_instance(build(problem, size, seed), configs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--problem", choices=["tsp", "cvrp"], default="tsp")
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
    kwargs = {name: {"checkpoint": args.checkpoint} for name in NEURAL} if args.checkpoint else {}
    seeds = range(args.seed, args.seed + args.n)
    rows: list[dict[str, Any]] = []

    if neural:  # warm up CUDA and load the checkpoint outside the timed region
        warm = build(args.problem, args.size, 0)
        for cfg in neural:
            name = parse_config(cfg)[0]
            get_solver(name, **kwargs.get(name, {})).solve(warm)
        for s in tqdm(seeds, desc="neural"):
            rows += run_instance(build(args.problem, args.size, s), neural, kwargs)

    if classic:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(_job, args.problem, args.size, s, classic) for s in seeds]
            for f in tqdm(as_completed(futures), total=len(futures), desc="classic"):
                rows += f.result()

    df = add_gaps(pd.DataFrame(rows))
    summary = summarize(df).drop(columns=["late_stops_%"])
    tag = f"synthetic_{args.problem}{args.size}{'_' + args.tag if args.tag else ''}"
    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "raw" / f"{tag}.parquet", index=False)
    md = args.out / f"{tag}.md"
    md.write_text(
        f"# Synthetic benchmark · {args.problem.upper()}-{args.size} · {args.n} instances\n\n"
        "Uniform instances, seeds not seen in training. "
        "Gap vs best solution found per instance.\n\n"
        + summary.to_markdown() + "\n",
        encoding="utf-8",
    )  # fmt: skip
    print(summary.to_string())
    print(f"\nwritten {md}")


if __name__ == "__main__":
    main()
