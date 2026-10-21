"""Fast local search (2-opt, or-opt, swap) to polish any solver's routes.

The solution is a *giant tour*: routes joined by depot separators, ``[0, a, b, 0, c, 0]``. Moves
work on that single array, so intra- and inter-route moves share one code path:

* **2-opt**     reverse a stretch inside one route. Asymmetric matrices are handled exactly with
                prefix sums of forward and backward arc costs (O(1) per candidate move).
* **or-opt**    move a run of 1-3 consecutive customers to any other position (same or another
                route), forwards or reversed, if the target route has capacity.
* **swap**      exchange two customers, within or across routes, if capacities allow.

Each pass applies the first improving move found and starts over, until no move improves the
travel time or ``max_moves`` is reached. Compiled with numba; the first call pays ~1-2 s of JIT.

Time windows are not modelled by the moves. When an instance has them, the polished solution is
kept only if it is not later than the input (see :func:`polish`).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from neural_routing.core.evaluate import evaluate
from neural_routing.core.instance import IntArray, Solution, VRPInstance
from neural_routing.solvers.base import Solver, get_solver, register

EPS = 1e-9
MAX_SEGMENT = 3


@njit(cache=True)
def _route_ids_and_loads(tour, q):  # type: ignore[no-untyped-def]
    """rid[p] = route that position p belongs to (a depot opens the next route)."""
    n = tour.size
    rid = np.empty(n, np.int64)
    loads = np.zeros(n, np.float64)
    r = -1
    for p in range(n):
        if tour[p] == 0:
            r += 1
        rid[p] = r
        loads[r] += q[tour[p]]
    return rid, loads


@njit(cache=True)
def _two_opt(tour, D):  # type: ignore[no-untyped-def]
    n = tour.size
    F = np.zeros(n)  # F[x] = cost of tour[0..x] driven forwards
    R = np.zeros(n)  # R[x] = cost of the same arcs driven backwards
    for t in range(n - 1):
        F[t + 1] = F[t] + D[tour[t], tour[t + 1]]
        R[t + 1] = R[t] + D[tour[t + 1], tour[t]]
    for i in range(1, n - 1):
        if tour[i] == 0:
            continue
        for k in range(i + 1, n - 1):
            if tour[k] == 0:
                break  # end of this route
            a, b, c, d = tour[i - 1], tour[i], tour[k], tour[k + 1]
            delta = (D[a, c] + D[b, d] - D[a, b] - D[c, d]) + (R[k] - R[i]) - (F[k] - F[i])
            if delta < -EPS:
                tour[i : k + 1] = tour[i : k + 1][::-1].copy()
                return True, delta
    return False, 0.0


@njit(cache=True)
def _or_opt(tour, D, q, cap):  # type: ignore[no-untyped-def]
    n = tour.size
    rid, loads = _route_ids_and_loads(tour, q)
    for L in range(1, MAX_SEGMENT + 1):
        for i in range(1, n - L):
            j = i + L - 1
            ok = True
            seg_load = 0.0
            for t in range(i, j + 1):
                if tour[t] == 0:
                    ok = False
                    break
                seg_load += q[tour[t]]
            if not ok:
                continue
            a, b, s0, sl = tour[i - 1], tour[j + 1], tour[i], tour[j]
            fwd = 0.0
            rev = 0.0
            for t in range(i, j):
                fwd += D[tour[t], tour[t + 1]]
                rev += D[tour[t + 1], tour[t]]
            gain = D[a, b] - D[a, s0] - D[sl, b]
            for p in range(n - 1):
                if i - 1 <= p <= j:
                    continue
                if rid[p] != rid[i] and loads[rid[p]] + seg_load > cap + EPS:
                    continue
                c, d = tour[p], tour[p + 1]
                ins_f = D[c, s0] + D[sl, d] - D[c, d]
                ins_r = D[c, sl] + D[s0, d] - D[c, d] + (rev - fwd)
                reverse = ins_r < ins_f
                delta = gain + (ins_r if reverse else ins_f)
                if delta < -EPS:
                    seg = tour[i : j + 1].copy()
                    if reverse:
                        seg = seg[::-1].copy()
                    rest = np.concatenate((tour[:i], tour[j + 1 :]))
                    at = p if p < i else p - L
                    new = np.concatenate((rest[: at + 1], seg, rest[at + 1 :]))
                    tour[:] = new
                    return True, delta
    return False, 0.0


@njit(cache=True)
def _swap(tour, D, q, cap):  # type: ignore[no-untyped-def]
    n = tour.size
    rid, loads = _route_ids_and_loads(tour, q)
    for u in range(1, n - 1):
        if tour[u] == 0:
            continue
        for v in range(u + 2, n - 1):  # adjacent pairs are covered by or-opt
            if tour[v] == 0:
                continue
            x, y = tour[u], tour[v]
            if rid[u] != rid[v]:
                if loads[rid[u]] - q[x] + q[y] > cap + EPS:
                    continue
                if loads[rid[v]] - q[y] + q[x] > cap + EPS:
                    continue
            pu, nu, pv, nv = tour[u - 1], tour[u + 1], tour[v - 1], tour[v + 1]
            old = D[pu, x] + D[x, nu] + D[pv, y] + D[y, nv]
            new = D[pu, y] + D[y, nu] + D[pv, x] + D[x, nv]
            delta = new - old
            if delta < -EPS:
                tour[u], tour[v] = y, x
                return True, delta
    return False, 0.0


@njit(cache=True)
def _improve(tour, D, q, cap, max_moves):  # type: ignore[no-untyped-def]
    moves = 0
    while moves < max_moves:
        found, _ = _two_opt(tour, D)
        if not found:
            found, _ = _or_opt(tour, D, q, cap)
        if not found:
            found, _ = _swap(tour, D, q, cap)
        if not found:
            break
        moves += 1
    return tour, moves


def split_giant_tour(tour: IntArray) -> list[list[int]]:
    routes: list[list[int]] = []
    current: list[int] = []
    for v in tour.tolist():
        if v == 0:
            if current:
                routes.append(current)
            current = []
        else:
            current.append(int(v))
    return routes


def improve_routes(
    instance: VRPInstance, routes: list[list[int]], max_moves: int = 100_000
) -> tuple[list[list[int]], int]:
    """Travel-time local search; returns (routes, number of applied moves)."""
    tour = np.array(Solution.from_lists(routes).giant_tour(), dtype=np.int64)
    cap = instance.capacity if math.isfinite(instance.capacity) else 1e300
    out, moves = _improve(
        tour, instance.durations.astype(np.float64), instance.demands.astype(np.float64),
        float(cap), max_moves,
    )  # fmt: skip
    return split_giant_tour(out), int(moves)


def polish(
    instance: VRPInstance, routes: list[list[int]], max_moves: int = 100_000
) -> list[list[int]]:
    """:func:`improve_routes`, but never returns a solution that is worse for the evaluator:
    with time windows the result must not add lateness, and travel time must not grow."""
    new, _ = improve_routes(instance, routes, max_moves)
    before = evaluate(instance, Solution.from_lists(routes))
    after = evaluate(instance, Solution.from_lists(new))
    worse_tw = after.lateness_s > before.lateness_s + 1e-6
    if worse_tw or after.travel_time_s > before.travel_time_s + 1e-6:
        return routes
    return new


class LocalSearchSolver(Solver):
    """Any registered solver followed by :func:`polish`."""

    base: str = ""

    def __init__(self, max_moves: int = 100_000, **base_kwargs: object) -> None:
        self.max_moves = max_moves
        self.inner = get_solver(self.base, **base_kwargs)

    def _solve(self, instance: VRPInstance, time_limit: float) -> list[list[int]]:
        # time_limit bounds the base solver; the search itself stops at a local optimum
        routes = [list(r) for r in self.inner.solve(instance, time_limit).routes]
        return polish(instance, routes, self.max_moves)


for _base in ("nearest_neighbor", "savings"):
    register(f"{_base}_ls")(type(f"{_base.title()}LS", (LocalSearchSolver,), {"base": _base}))
