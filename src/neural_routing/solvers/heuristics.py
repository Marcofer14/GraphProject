"""Construction heuristics: the "no optimisation" end of the quality/speed curve."""

from __future__ import annotations

import numpy as np

from neural_routing.core.instance import VRPInstance
from neural_routing.solvers.base import Solver, register


@register("reference")
class ReferenceSolver(Solver):
    """Replays the routes stored in the instance (the real Amazon driver's sequence)."""

    def _solve(self, instance: VRPInstance, time_limit: float) -> list[list[int]]:
        if instance.reference_routes is None:
            raise ValueError(f"{instance.name} has no reference routes")
        return [list(r) for r in instance.reference_routes]


@register("nearest_neighbor")
class NearestNeighborSolver(Solver):
    """Always drive to the closest unvisited customer that still fits; return to depot when none."""

    def _solve(self, instance: VRPInstance, time_limit: float) -> list[list[int]]:
        D, q, cap = instance.durations, instance.demands, instance.capacity
        unvisited = np.ones(instance.num_nodes, dtype=bool)
        unvisited[0] = False
        routes: list[list[int]] = []
        while unvisited.any():
            route, load, cur = [], 0.0, 0
            while True:
                cand = unvisited & (q + load <= cap)
                if not cand.any():
                    break
                nxt = int(np.argmin(np.where(cand, D[cur], np.inf)))
                route.append(nxt)
                unvisited[nxt] = False
                load += q[nxt]
                cur = nxt
            if not route:  # a single customer above capacity: serve it alone
                nxt = int(np.flatnonzero(unvisited)[0])
                route, unvisited[nxt] = [nxt], False
            routes.append(route)
        return routes


@register("savings")
class SavingsSolver(Solver):
    """Clarke-Wright parallel savings for asymmetric matrices.

    Joining route A (ending at i) with route B (starting at j) saves D[i,0] + D[0,j] - D[i,j].
    Only the ``k`` nearest successors of each node are considered, which keeps large pooled
    instances tractable with negligible quality loss.
    """

    def __init__(self, k: int = 40) -> None:
        self.k = k

    def _solve(self, instance: VRPInstance, time_limit: float) -> list[list[int]]:
        D, q, cap = instance.durations, instance.demands, instance.capacity
        n = instance.num_nodes
        cust = np.arange(1, n)
        k = min(self.k, n - 2)
        if k <= 0:
            return [[int(c)] for c in cust]

        sub = D[1:, 1:].copy()
        np.fill_diagonal(sub, np.inf)
        nbr = np.argpartition(sub, k - 1, axis=1)[:, :k] + 1  # (n-1, k) successor candidates
        i = np.repeat(cust, k)
        j = nbr.ravel()
        saving = D[i, 0] + D[0, j] - D[i, j]
        order = np.argsort(-saving, kind="stable")

        route_of = {int(c): int(c) for c in cust}  # node -> route id
        routes: dict[int, list[int]] = {int(c): [int(c)] for c in cust}
        load = {int(c): float(q[c]) for c in cust}
        for a, b, s in zip(i[order], j[order], saving[order], strict=True):
            if s <= 0:
                break
            ra, rb = route_of[int(a)], route_of[int(b)]
            if ra == rb or routes[ra][-1] != a or routes[rb][0] != b:
                continue
            if load[ra] + load[rb] > cap:
                continue
            routes[ra].extend(routes[rb])
            load[ra] += load.pop(rb)
            for v in routes.pop(rb):
                route_of[v] = ra
        return _merge_to_fleet(list(routes.values()), instance)


def _merge_to_fleet(routes: list[list[int]], instance: VRPInstance) -> list[list[int]]:
    """Neighbour-restricted savings can leave fragments; join the best-saving pairs (full matrix)
    until the fleet size is met or no join fits the capacity."""
    D, q, cap = instance.durations, instance.demands, instance.capacity
    loads = [float(q[r].sum()) for r in routes]
    while len(routes) > instance.num_vehicles:
        best, pair = -np.inf, None
        for a, ra in enumerate(routes):
            for b, rb in enumerate(routes):
                if a != b and loads[a] + loads[b] <= cap:
                    s = D[ra[-1], 0] + D[0, rb[0]] - D[ra[-1], rb[0]]
                    if s > best:
                        best, pair = s, (a, b)
        if pair is None:
            break
        a, b = pair
        routes[a] = routes[a] + routes[b]
        loads[a] += loads[b]
        del routes[b], loads[b]
    return routes
