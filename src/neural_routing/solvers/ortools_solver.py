"""Google OR-Tools routing (guided local search): the industry-standard baseline."""

from __future__ import annotations

import math

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from neural_routing.core.instance import VRPInstance
from neural_routing.solvers.base import INT_INF, Solver, fleet_size, register

HORIZON = 7 * 24 * 3600


@register("ortools")
class ORToolsSolver(Solver):
    """Minimises travel time; late arrivals cost ``late_penalty`` per second (soft windows)."""

    def __init__(self, late_penalty: int = 10) -> None:
        self.late_penalty = late_penalty

    def _solve(self, instance: VRPInstance, time_limit: float) -> list[list[int]]:
        n = instance.num_nodes
        vehicles = fleet_size(instance)
        D = np.rint(instance.durations).astype(np.int64)
        service = np.rint(instance.service_times).astype(np.int64)
        transit = D + service[:, None]  # leaving i includes serving i

        manager = pywrapcp.RoutingIndexManager(n, vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def arc(i: int, j: int) -> int:
            return int(D[manager.IndexToNode(i), manager.IndexToNode(j)])

        def time_cb(i: int, j: int) -> int:
            return int(transit[manager.IndexToNode(i), manager.IndexToNode(j)])

        def demand_cb(i: int) -> int:
            return int(instance.demands[manager.IndexToNode(i)])

        routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(arc))

        time_idx = routing.RegisterTransitCallback(time_cb)
        routing.AddDimension(time_idx, HORIZON, HORIZON, True, "time")  # slack = waiting
        time_dim = routing.GetDimensionOrDie("time")
        for node in range(1, n):
            early, late = instance.time_windows[node]
            cumul = time_dim.CumulVar(manager.NodeToIndex(node))
            cumul.SetMin(int(early))
            if math.isfinite(late) and late < INT_INF:
                time_dim.SetCumulVarSoftUpperBound(
                    manager.NodeToIndex(node), int(late), self.late_penalty
                )

        if math.isfinite(instance.capacity):
            cap_idx = routing.RegisterUnaryTransitCallback(demand_cb)
            routing.AddDimensionWithVehicleCapacity(
                cap_idx, 0, [int(instance.capacity)] * vehicles, True, "load"
            )

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        params.time_limit.FromMilliseconds(max(1, int(time_limit * 1000)))

        sol = routing.SolveWithParameters(params)
        if sol is None:
            raise RuntimeError(f"OR-Tools found no solution for {instance.name}")
        routes = []
        for v in range(vehicles):
            idx, route = sol.Value(routing.NextVar(routing.Start(v))), []
            while not routing.IsEnd(idx):
                route.append(manager.IndexToNode(idx))
                idx = sol.Value(routing.NextVar(idx))
            routes.append(route)
        return routes
