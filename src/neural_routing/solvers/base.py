"""Common solver interface. Classic heuristics, OR-Tools, PyVRP and (later) neural policies all
implement :class:`Solver`, so benchmarks, the API and the UI treat them identically."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from neural_routing.core.instance import Solution, VRPInstance

INT_INF = 10**9  # "no deadline" for integer-based solvers (~31 years in seconds)


class Solver(ABC):
    name: str = "solver"

    def solve(self, instance: VRPInstance, time_limit: float = 1.0) -> Solution:
        t0 = time.perf_counter()
        routes = self._solve(instance, time_limit)
        return Solution.from_lists(routes, solver=self.name, runtime_s=time.perf_counter() - t0)

    @abstractmethod
    def _solve(self, instance: VRPInstance, time_limit: float) -> list[list[int]]: ...


def fleet_size(instance: VRPInstance) -> int:
    """Vehicles offered to fleet-limited solvers: the real fleet, or more if demand requires it."""
    needed = (
        math.ceil(float(instance.demands.sum()) / instance.capacity) if instance.capacity else 1
    )
    return max(instance.num_vehicles, needed)


def to_int(x: float) -> int:
    return INT_INF if not math.isfinite(x) else round(x)


_REGISTRY: dict[str, Callable[..., Solver]] = {}


def register(name: str) -> Callable[[type[Solver]], type[Solver]]:
    def deco(cls: type[Solver]) -> type[Solver]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_solver(name: str, **kwargs: object) -> Solver:
    if name not in _REGISTRY:
        raise KeyError(f"unknown solver {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_solvers() -> list[str]:
    return sorted(_REGISTRY)
