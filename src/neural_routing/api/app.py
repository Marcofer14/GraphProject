"""HTTP API over the solvers: list instances, solve one with any solver, compare several.

    uv run nr-api                     # http://127.0.0.1:8000/docs
    NR_DATA_ROOT=data/processed NR_CHECKPOINTS=checkpoints uv run nr-api

Neural solvers take a ``model``: the *name* of a file in the checkpoints folder, never a path
(loading an arbitrary PyTorch checkpoint can execute code).
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from neural_routing.api.schemas import (
    CompareOut,
    CompareRequest,
    InstanceDetail,
    InstanceSummary,
    ModelInfo,
    SolutionOut,
    SolveRequest,
    SolverInfo,
    Stop,
)
from neural_routing.core import VRPInstance, evaluate
from neural_routing.data.amazon import AmazonDataset
from neural_routing.eval.benchmark import NEURAL
from neural_routing.solvers import available_solvers, get_solver

DESCRIPTIONS = {
    "reference": "The real Amazon driver's sequence",
    "nearest_neighbor": "Greedy: always the closest unvisited stop",
    "nearest_neighbor_ls": "Nearest neighbour + 2-opt / or-opt / swap",
    "savings": "Clarke-Wright savings heuristic",
    "savings_ls": "Clarke-Wright + 2-opt / or-opt / swap",
    "ortools": "Google OR-Tools, guided local search",
    "pyvrp": "PyVRP hybrid genetic search (near-optimal reference)",
    "attention": "Attention model + POMO, multi-start x8 augmentation",
    "attention_ls": "Attention model + POMO + local search (the hybrid)",
    "attention_greedy": "Attention model, single greedy rollout",
}


def solver_kind(name: str) -> str:
    if name == "reference":
        return "reference"
    return "neural" if name in NEURAL else "classic"


def stop_out(inst: VRPInstance, i: int) -> Stop:
    zones = inst.meta.get("zones", ())
    zone = zones[i] if i < len(zones) and isinstance(zones[i], str) else None  # station: NaN
    start, end = (float(t) for t in inst.time_windows[i])
    windowed = math.isfinite(end)
    return Stop(
        node=i,
        stop_id=inst.node_ids[i] if inst.node_ids else str(i),
        lat=float(inst.coords[i, 0]),
        lng=float(inst.coords[i, 1]),
        zone=zone,
        demand=float(inst.demands[i] / inst.capacity) if math.isfinite(inst.capacity) else 0.0,
        service_s=float(inst.service_times[i]),
        tw_start_s=start if windowed else None,
        tw_end_s=end if windowed else None,
    )


def create_app(data_root: Path | None = None, checkpoint_dir: Path | None = None) -> FastAPI:
    data_root = data_root or Path(os.environ.get("NR_DATA_ROOT", "data/processed"))
    ckpt_dir = checkpoint_dir or Path(os.environ.get("NR_CHECKPOINTS", "checkpoints"))
    app = FastAPI(title="Neural last-mile routing", version="0.4.0")
    app.add_middleware(  # the Next.js front end runs on another port during development
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @lru_cache(maxsize=4)
    def dataset(split: str) -> AmazonDataset:
        try:
            return AmazonDataset(split, root=data_root)
        except FileNotFoundError as exc:
            raise HTTPException(404, f"split {split!r} not processed") from exc

    def instance(split: str, route_id: str) -> VRPInstance:
        ds = dataset(split)
        if route_id not in ds.routes.index:
            raise HTTPException(404, f"route {route_id!r} not in split {split!r}")
        return ds.route_instance(route_id)

    def model_path(name: str | None) -> Path:
        available = {p.stem: p for p in ckpt_dir.glob("*.pt") if not p.stem.endswith("_last")}
        if name is None or name not in available:
            raise HTTPException(404, f"model {name!r} not found; available: {sorted(available)}")
        return available[name]

    def solve(inst: VRPInstance, solver: str, time_limit: float, model: str | None) -> SolutionOut:
        if solver not in available_solvers():
            raise HTTPException(400, f"unknown solver {solver!r}")
        kwargs: dict[str, Any] = {}
        if solver in NEURAL:
            kwargs["checkpoint"] = model_path(model)
        elif solver == "reference" and inst.reference_routes is None:
            raise HTTPException(400, "this route has no recorded driver sequence")
        try:
            sol = get_solver(solver, **kwargs).solve(inst, time_limit=time_limit)
        except HTTPException:
            raise
        except Exception as exc:  # surface solver failures as a clean API error
            raise HTTPException(422, f"{solver} failed: {exc}") from exc
        m = evaluate(inst, sol)
        latlng = inst.coords
        paths = [
            [(float(latlng[v, 1]), float(latlng[v, 0])) for v in (0, *route, 0)]
            for route in sol.routes
        ]
        return SolutionOut(
            solver=solver,
            model=model if solver in NEURAL else None,
            routes=[list(r) for r in sol.routes],
            paths=paths,
            travel_time_s=m.travel_time_s,
            route_duration_s=m.route_duration_s,
            num_routes=m.num_routes,
            tw_violations=m.tw_violations,
            lateness_s=m.lateness_s,
            feasible=m.is_feasible,
            runtime_s=m.runtime_s,
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "cuda": torch.cuda.is_available(), "data_root": str(data_root)}

    @app.get("/solvers", response_model=list[SolverInfo])
    def solvers() -> list[SolverInfo]:
        return [
            SolverInfo(name=s, kind=solver_kind(s), description=DESCRIPTIONS.get(s, ""))
            for s in available_solvers()
        ]

    @app.get("/models", response_model=list[ModelInfo])
    def models() -> list[ModelInfo]:
        out = []
        for p in sorted(ckpt_dir.glob("*.pt")):
            if p.stem.endswith("_last"):
                continue
            meta = torch.load(p, map_location="cpu", weights_only=False)
            out.append(ModelInfo(name=p.stem, problem=str(meta.get("problem", "?")),
                                 num_customers=int(meta.get("num_customers", 0)),
                                 epoch=int(meta.get("epoch", -1))))  # fmt: skip
        return out

    @app.get("/instances", response_model=list[InstanceSummary])
    def instances(
        split: str = "evaluation", limit: int = 50, offset: int = 0
    ) -> list[InstanceSummary]:
        routes = dataset(split).routes.iloc[offset : offset + min(limit, 500)]
        return [
            InstanceSummary(
                route_id=str(rid),
                station=str(r["station_code"]),
                date=str(r["date"]),
                num_stops=int(r["n_nodes"]) - 1,
                route_score=r["route_score"] if isinstance(r["route_score"], str) else None,
            )
            for rid, r in routes.iterrows()
        ]

    @app.get("/instances/{split}/{route_id}", response_model=InstanceDetail)
    def instance_detail(split: str, route_id: str) -> InstanceDetail:
        inst = instance(split, route_id)
        stops = [stop_out(inst, i) for i in range(inst.num_nodes)]
        return InstanceDetail(
            route_id=route_id,
            station=str(inst.meta.get("station", "")),
            date=str(inst.meta.get("date", "")),
            capacity_cm3=inst.capacity,
            stops=stops,
            driver_route=list(inst.reference_routes[0]) if inst.reference_routes else None,
        )

    @app.post("/solve", response_model=SolutionOut)
    def solve_one(req: SolveRequest) -> SolutionOut:
        return solve(instance(req.split, req.route_id), req.solver, req.time_limit, req.model)

    @app.post("/compare", response_model=CompareOut)
    def compare(req: CompareRequest) -> CompareOut:
        inst = instance(req.split, req.route_id)
        sols = [solve(inst, s, req.time_limit, req.model) for s in req.solvers]
        best = min(sols, key=lambda s: s.travel_time_s)
        for s in sols:
            s.gap_pct = 100 * (s.travel_time_s / best.travel_time_s - 1)
        return CompareOut(route_id=req.route_id, best=best.solver, solutions=sols)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("neural_routing.api.app:app", host="127.0.0.1", port=8000, reload=False)
