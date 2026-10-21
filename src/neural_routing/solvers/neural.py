"""The trained attention policy behind the common :class:`Solver` interface.

The policy works on unit-square coordinates and Euclidean distances, so coordinates are rescaled
(keeping the aspect ratio) and demands divided by capacity. Routes are decoded greedily from every
POMO start (and the 8 symmetric augmentations); the best tour *by the instance's own matrix* wins,
which keeps the result honest when the matrix is not Euclidean.
"""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import numpy as np
import torch

from neural_routing.core.evaluate import evaluate
from neural_routing.core.instance import Solution, VRPInstance
from neural_routing.ml.env import Batch, augment8
from neural_routing.ml.model import AttentionPolicy
from neural_routing.solvers.base import Solver, register

DEFAULT_CHECKPOINTS = Path("checkpoints")


@cache
def _load(path: str, device: str) -> AttentionPolicy:
    from neural_routing.ml.train import load_policy

    return load_policy(path, device)[0]


def split_routes(actions: list[int]) -> list[list[int]]:
    routes: list[list[int]] = [[]]
    for a in actions:
        if a == 0:
            if routes[-1]:
                routes.append([])
        else:
            routes[-1].append(a)
    return [r for r in routes if r]


def to_batch(instance: VRPInstance, device: str) -> Batch:
    xy = instance.coords - instance.coords.min(0)
    xy = xy / max(float(xy.max()), 1e-9)
    tsp = not math.isfinite(instance.capacity)
    demand = np.zeros(instance.num_nodes) if tsp else instance.demands / instance.capacity
    return Batch(
        torch.tensor(xy[None], dtype=torch.float32, device=device),
        torch.tensor(demand[None], dtype=torch.float32, device=device),
        tsp,
    )


@register("attention")
class AttentionSolver(Solver):
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        augment: bool = True,
        multistart: bool = True,
        device: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.augment = augment
        self.multistart = multistart
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _checkpoint_for(self, instance: VRPInstance) -> str:
        if self.checkpoint:
            return str(self.checkpoint)
        problem = "tsp" if not math.isfinite(instance.capacity) else "cvrp"
        found = sorted(DEFAULT_CHECKPOINTS.glob(f"{problem}*_pomo.pt"))
        if not found:
            raise FileNotFoundError(f"no {problem} checkpoint in {DEFAULT_CHECKPOINTS}/")

        def size_gap(p: Path) -> int:  # checkpoints are named like cvrp50_pomo.pt
            return abs(int(p.stem.removeprefix(problem).split("_")[0]) - instance.num_customers)

        return str(min(found, key=size_gap))

    @torch.no_grad()
    def _solve(self, instance: VRPInstance, time_limit: float) -> list[list[int]]:
        policy = _load(self._checkpoint_for(instance), self.device)
        batch = to_batch(instance, self.device)
        if self.augment:
            batch = augment8(batch)
        P = instance.num_customers if self.multistart else 1
        _, _, actions = policy(batch, P, decode="greedy", forced_starts=self.multistart)
        candidates = actions.reshape(-1, actions.shape[-1]).tolist()
        # score each distinct candidate with the real matrix and keep the best
        best, best_cost = None, math.inf
        for acts in {tuple(a) for a in candidates}:
            routes = split_routes(list(acts))
            cost = evaluate(instance, Solution.from_lists(routes)).travel_time_s
            if cost < best_cost:
                best, best_cost = routes, cost
        assert best is not None
        return best


@register("attention_greedy")
class AttentionGreedySolver(AttentionSolver):
    """Single greedy rollout, no augmentation: the fastest neural setting."""

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None) -> None:
        super().__init__(checkpoint, augment=False, multistart=False, device=device)
