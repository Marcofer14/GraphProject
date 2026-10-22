"""Cluster-first, route-second: plan a whole station-day with a single-van route solver.

1. **Cluster** the stops into one group per van with a capacitated k-means on planar km
   coordinates (k-means++ seeding; each round assigns stops in order of *regret* - how much worse
   their second-best van is - to the nearest van with room left, then moves the centres).
2. **Route** each group as its own one-van instance with any registered solver (the neural hybrid
   by default), on the instance's own travel-time sub-matrix.
3. **Polish** the whole day with the multi-route local search (or-opt / swap move stops between
   vans when capacity allows), bounded by ``max_moves``.

Group sizes follow the real fleet (``fleet_size``), so the comparison with the drivers' day keeps
the number of vans fixed.
"""

from __future__ import annotations

import math

import numpy as np

from neural_routing.core.instance import VRPInstance
from neural_routing.solvers.base import Solver, fleet_size, get_solver, register
from neural_routing.solvers.local_search import improve_routes
from neural_routing.solvers.neural import planar_km


def capacitated_kmeans(
    xy: np.ndarray, demand: np.ndarray, k: int, capacity: float, iters: int = 15, seed: int = 0
) -> np.ndarray:
    """Group label per point (0..k-1) with the total demand of every group <= capacity.

    Raises ValueError when the demand cannot fit in k groups at all.
    """
    if demand.sum() > k * capacity + 1e-9:
        raise ValueError(f"total demand {demand.sum():.3g} exceeds {k} x capacity {capacity:.3g}")
    rng = np.random.default_rng(seed)
    n = len(xy)
    centres = [xy[rng.integers(n)]]
    for _ in range(1, k):  # k-means++ seeding
        d2 = np.min([((xy - c) ** 2).sum(1) for c in centres], axis=0)
        centres.append(xy[rng.choice(n, p=d2 / d2.sum()) if d2.sum() > 0 else rng.integers(n)])
    C = np.array(centres)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        dist = np.sqrt(((xy[:, None] - C[None]) ** 2).sum(-1))  # (n, k)
        ranked = np.sort(dist, axis=1)
        regret = ranked[:, 1] - ranked[:, 0] if k > 1 else np.zeros(n)
        load = np.zeros(k)
        new = np.full(n, -1)
        for i in np.argsort(-regret, kind="stable"):  # hardest decisions first
            for c in np.argsort(dist[i]):
                if load[c] + demand[i] <= capacity + 1e-9:
                    new[i], load[c] = c, load[c] + demand[i]
                    break
            if new[i] < 0:  # nothing fits: least loaded group (only possible with huge stops)
                c = int(np.argmin(load))
                new[i], load[c] = c, load[c] + demand[i]
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            if (labels == c).any():
                C[c] = xy[labels == c].mean(0)
    return labels


def sub_instance(instance: VRPInstance, members: np.ndarray) -> VRPInstance:
    """One-van instance with the depot plus ``members`` (customer node ids of ``instance``)."""
    idx = np.concatenate([[0], members])
    refs = None
    if instance.reference_routes:
        member_set = set(members.tolist())
        local = {int(v): i + 1 for i, v in enumerate(members)}
        for route in instance.reference_routes:  # the driver route of this group, if one matches
            if set(route) == member_set:
                refs = (tuple(local[v] for v in route),)
    return VRPInstance(
        name=f"{instance.name}/g{len(members)}",
        coords=instance.coords[idx],
        durations=instance.durations[np.ix_(idx, idx)],
        demands=instance.demands[idx],
        service_times=instance.service_times[idx],
        time_windows=instance.time_windows[idx],
        capacity=math.inf,
        num_vehicles=1,
        node_ids=tuple(instance.node_ids[i] for i in idx) if instance.node_ids else (),
        reference_routes=refs,
        meta={**instance.meta},
    )


class ClusterFirstSolver(Solver):
    inner: str = "attention_ls"

    def __init__(self, max_moves: int = 3000, seed: int = 0, **inner_kwargs: object) -> None:
        self.max_moves = max_moves
        self.seed = seed
        self.route_solver = get_solver(self.inner, **inner_kwargs)

    def _solve(self, instance: VRPInstance, time_limit: float) -> list[list[int]]:
        k = fleet_size(instance)
        customers = np.arange(1, instance.num_nodes)
        labels = capacitated_kmeans(
            planar_km(instance)[1:], instance.demands[1:], k, instance.capacity, seed=self.seed
        )
        per_group_limit = max(0.1, time_limit / k)
        routes: list[list[int]] = []
        for g in range(k):
            members = customers[labels == g]
            if len(members) == 0:
                continue
            if len(members) == 1:
                routes.append([int(members[0])])
                continue
            sub = sub_instance(instance, members)
            sub_sol = self.route_solver.solve(sub, time_limit=per_group_limit)
            for r in sub_sol.routes:  # back to the day's node ids
                routes.append([int(members[v - 1]) for v in r])
        if self.max_moves:
            routes, _ = improve_routes(instance, routes, max_moves=self.max_moves)
        return routes


@register("cluster_attention_ls")
class ClusterAttention(ClusterFirstSolver):
    """Day plan: capacitated k-means + the neural hybrid per van + cross-van local search."""

    inner = "attention_ls"


@register("cluster_savings_ls")
class ClusterSavings(ClusterFirstSolver):
    """Control: same clustering and polish, savings + local search per van instead of the net."""

    inner = "savings_ls"
