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
from neural_routing.ml.generators import normalise
from neural_routing.ml.model import AttentionPolicy
from neural_routing.solvers.base import Solver, register

DEFAULT_CHECKPOINTS = Path("checkpoints")


@cache
def _load(path: str, device: str) -> tuple[AttentionPolicy, str]:
    """The policy and the coordinate normalisation it was trained with (old checkpoints: all)."""
    from neural_routing.ml.train import load_policy

    policy, ckpt = load_policy(path, device)
    data = (ckpt.get("train_config") or {}).get("data") or {}
    return policy, str(data.get("normalisation", "all"))


def split_routes(actions: list[int]) -> list[list[int]]:
    routes: list[list[int]] = [[]]
    for a in actions:
        if a == 0:
            if routes[-1]:
                routes.append([])
        else:
            routes[-1].append(a)
    return [r for r in routes if r]


def planar_km(instance: VRPInstance) -> np.ndarray:
    """Coordinates on a plane where both axes have the same unit.

    Real instances carry (lat, lng) in degrees; a degree of longitude is cos(lat) shorter than a
    degree of latitude, so they are projected (equirectangular) to km. Synthetic ones are planar.
    """
    if instance.meta.get("coord_system") != "latlng":
        return np.asarray(instance.coords, dtype=np.float64)
    lat, lng = instance.coords[:, 0], instance.coords[:, 1]
    lat0 = np.radians(lat.mean())
    return np.column_stack([lng * 111.32 * np.cos(lat0), lat * 110.574])


def seconds_per_unit(instance: VRPInstance, locs: np.ndarray) -> float:
    """Least-squares speed: travel seconds per unit of normalised Euclidean distance.

    Fits ``durations ~ s * euclid(locs)`` through the origin, so times from the real matrix
    become comparable with the distances the policy sees.
    """
    E = np.linalg.norm(locs[:, None] - locs[None], axis=-1)
    off = ~np.eye(len(locs), dtype=bool)
    num, den = float((instance.durations[off] * E[off]).sum()), float((E[off] ** 2).sum())
    return num / den if den > 0 else 1.0


def to_batch(
    instance: VRPInstance, device: str, normalisation: str = "all", with_time: bool = False
) -> Batch:
    xy = torch.tensor(planar_km(instance)[None], dtype=torch.float32)
    locs, _ = normalise(xy, normalisation)
    tsp = not math.isfinite(instance.capacity)
    demand = np.zeros(instance.num_nodes) if tsp else instance.demands / instance.capacity
    service = tw = None
    if with_time:
        s = seconds_per_unit(instance, locs[0].double().numpy())
        service = torch.tensor(instance.service_times[None] / s, dtype=torch.float32, device=device)
        tw = torch.tensor(instance.time_windows[None] / s, dtype=torch.float32, device=device)
    return Batch(
        locs.to(device),
        torch.tensor(demand[None], dtype=torch.float32, device=device),
        tsp,
        service,
        tw,
    )


@register("attention")
class AttentionSolver(Solver):
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        augment: bool = True,
        multistart: bool = True,
        device: str | None = None,
        polish_top: int = 0,
        tw_penalty: float = 10.0,
    ) -> None:
        self.checkpoint = checkpoint
        self.augment = augment
        self.multistart = multistart
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.polish_top = polish_top  # local-search the k best decoded candidates
        self.tw_penalty = tw_penalty  # seconds of travel one second of lateness is worth

    def _objective(self, instance: VRPInstance, routes: list[list[int]]) -> float:
        m = evaluate(instance, Solution.from_lists(routes))
        return m.travel_time_s + self.tw_penalty * m.lateness_s

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
        policy, normalisation = _load(self._checkpoint_for(instance), self.device)
        timed = policy.cfg.time_features and instance.has_time_windows
        batch = to_batch(instance, self.device, normalisation, with_time=timed)
        if self.augment:
            batch = augment8(batch)
        P = instance.num_customers if self.multistart else 1
        _, _, actions = policy(batch, P, decode="greedy", forced_starts=self.multistart)
        candidates = actions.reshape(-1, actions.shape[-1]).tolist()
        # score each distinct candidate on the real matrix: travel + penalty x lateness
        scored = []
        for acts in {tuple(a) for a in candidates}:
            routes = split_routes(list(acts))
            scored.append((self._objective(instance, routes), routes))
        scored.sort(key=lambda c: c[0])
        if self.polish_top:
            from neural_routing.solvers.local_search import polish

            polished = [polish(instance, r) for _, r in scored[: self.polish_top]]
            return min(polished, key=lambda r: self._objective(instance, r))
        return scored[0][1]


@register("attention_ls")
class AttentionLSSolver(AttentionSolver):
    """Multi-start + augmentation, then 2-opt / or-opt / swap on the 8 best candidates."""

    def __init__(
        self, checkpoint: str | Path | None = None, device: str | None = None, polish_top: int = 8
    ) -> None:
        super().__init__(checkpoint, device=device, polish_top=polish_top)


@register("attention_greedy")
class AttentionGreedySolver(AttentionSolver):
    """Single greedy rollout, no augmentation: the fastest neural setting."""

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None) -> None:
        super().__init__(checkpoint, augment=False, multistart=False, device=device)
