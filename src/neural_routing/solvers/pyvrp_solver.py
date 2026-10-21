"""PyVRP hybrid genetic search: close to state of the art, used as "best known" reference."""

from __future__ import annotations

import math

import numpy as np
import pyvrp
from pyvrp.stop import MaxRuntime

from neural_routing.core.instance import VRPInstance
from neural_routing.solvers.base import Solver, fleet_size, register, to_int


@register("pyvrp")
class PyVRPSolver(Solver):
    """Distance = travel time, so PyVRP optimises the same objective we evaluate.

    PyVRP treats time windows as hard constraints (violations are penalised as "time warp"
    during search). Infeasible windows still yield a solution, which our evaluator scores.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _solve(self, instance: VRPInstance, time_limit: float) -> list[list[int]]:
        n = instance.num_nodes
        D = np.rint(instance.durations).astype(np.int64)
        np.fill_diagonal(D, 0)  # PyVRP rejects non-zero self distances
        has_cap = math.isfinite(instance.capacity)
        locations = [pyvrp.Location(float(x), float(y)) for x, y in instance.coords]
        depots = [pyvrp.Depot(0)]
        clients = [
            pyvrp.Client(
                i,
                delivery=[round(instance.demands[i])] if has_cap else [],
                service_duration=round(instance.service_times[i]),
                tw_early=round(instance.time_windows[i, 0]),
                tw_late=to_int(instance.time_windows[i, 1]),
            )
            for i in range(1, n)
        ]
        vehicles = [
            pyvrp.VehicleType(
                fleet_size(instance), capacity=[round(instance.capacity)] if has_cap else []
            )
        ]
        data = pyvrp.ProblemData(locations, clients, depots, vehicles, [D], [D])  # type: ignore[list-item]
        res = pyvrp.solve(data, stop=MaxRuntime(time_limit), seed=self.seed, display=False)
        # client activity idx is 0-based among clients; our node index is idx + 1 (single depot)
        return [[a.idx + 1 for a in route if a.is_client()] for route in res.best.routes()]
