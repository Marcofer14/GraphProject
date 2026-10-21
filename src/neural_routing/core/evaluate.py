"""Single source of truth for solution cost and feasibility.

Every solver is scored here, never by its own internal objective, so numbers are comparable.
Time windows are *soft*: arriving early means waiting, arriving late is allowed but measured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from neural_routing.core.instance import Solution, VRPInstance


@dataclass(frozen=True)
class Metrics:
    travel_time_s: float  # sum of driving time, the main objective
    route_duration_s: float  # driving + service + waiting, summed over vehicles
    num_routes: int
    tw_violations: int  # customers served after their latest time
    lateness_s: float  # total seconds late
    capacity_excess: float  # total load above capacity, summed over routes
    missing: int  # customers never visited
    duplicated: int  # extra visits to already served customers
    runtime_s: float

    @property
    def is_complete(self) -> bool:
        return self.missing == 0 and self.duplicated == 0

    @property
    def is_feasible(self) -> bool:
        return self.is_complete and self.tw_violations == 0 and self.capacity_excess <= 1e-9

    def as_dict(self) -> dict[str, float | int | bool]:
        d: dict[str, float | int | bool] = dict(asdict(self))
        d["is_complete"] = self.is_complete
        d["is_feasible"] = self.is_feasible
        return d


def evaluate(instance: VRPInstance, solution: Solution) -> Metrics:
    D = instance.durations
    tw = instance.time_windows
    visits = np.zeros(instance.num_nodes, dtype=np.int64)
    travel = duration = lateness = excess = 0.0
    late_count = 0

    for route in solution.routes:
        t = 0.0
        load = 0.0
        prev = 0
        for node in route:
            if not 0 < node < instance.num_nodes:
                raise ValueError(f"route contains invalid node {node}")
            visits[node] += 1
            leg = D[prev, node]
            travel += leg
            t = max(t + leg, tw[node, 0])  # wait if early
            if t > tw[node, 1] + 1e-6:
                late_count += 1
                lateness += t - tw[node, 1]
            t += instance.service_times[node]
            load += instance.demands[node]
            prev = node
        travel += D[prev, 0]
        duration += t + D[prev, 0]
        excess += max(0.0, load - instance.capacity)

    customers = visits[1:]
    return Metrics(
        travel_time_s=float(travel),
        route_duration_s=float(duration),
        num_routes=solution.num_routes,
        tw_violations=late_count,
        lateness_s=float(lateness),
        capacity_excess=float(excess),
        missing=int((customers == 0).sum()),
        duplicated=int(np.clip(customers - 1, 0, None).sum()),
        runtime_s=solution.runtime_s,
    )
