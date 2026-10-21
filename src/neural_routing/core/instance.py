"""Problem and solution representation shared by every solver (classic or neural).

Conventions
-----------
* Node 0 is the depot; customers are nodes 1..N.
* All times are seconds relative to the vehicle departure from the depot.
* ``durations[i, j]`` is the (possibly asymmetric) travel time from i to j.
* A customer without a time window has ``tw = [0, inf]``.
* Routes in a :class:`Solution` list customer nodes only; depot start/end are implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class VRPInstance:
    """Capacitated VRP with (soft) time windows over a travel-time matrix."""

    name: str
    coords: FloatArray  # (N+1, 2) lat/lng in degrees, or planar x/y for synthetic data
    durations: FloatArray  # (N+1, N+1) seconds
    demands: FloatArray  # (N+1,) same unit as capacity; demands[0] == 0
    service_times: FloatArray  # (N+1,) seconds
    time_windows: FloatArray  # (N+1, 2) [earliest, latest] seconds
    capacity: float
    num_vehicles: int = 1
    node_ids: tuple[str, ...] = ()
    reference_routes: tuple[tuple[int, ...], ...] | None = None  # e.g. what the real driver did
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.num_nodes
        expected = {
            "coords": (n, 2),
            "durations": (n, n),
            "demands": (n,),
            "service_times": (n,),
            "time_windows": (n, 2),
        }
        for attr, shape in expected.items():
            if getattr(self, attr).shape != shape:
                raise ValueError(f"{attr} has shape {getattr(self, attr).shape}, expected {shape}")
        if self.node_ids and len(self.node_ids) != n:
            raise ValueError("node_ids must have one entry per node")

    @property
    def num_nodes(self) -> int:
        return int(self.demands.shape[0])

    @property
    def num_customers(self) -> int:
        return self.num_nodes - 1

    @property
    def has_time_windows(self) -> bool:
        return bool(np.isfinite(self.time_windows[1:, 1]).any())

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            name=self.name,
            coords=self.coords,
            durations=self.durations,
            demands=self.demands,
            service_times=self.service_times,
            time_windows=self.time_windows,
            capacity=self.capacity,
            num_vehicles=self.num_vehicles,
            node_ids=np.array(self.node_ids, dtype=str),
        )

    @classmethod
    def load(cls, path: Path) -> VRPInstance:
        z = np.load(path, allow_pickle=False)
        return cls(
            name=str(z["name"]),
            coords=z["coords"],
            durations=z["durations"],
            demands=z["demands"],
            service_times=z["service_times"],
            time_windows=z["time_windows"],
            capacity=float(z["capacity"]),
            num_vehicles=int(z["num_vehicles"]),
            node_ids=tuple(str(s) for s in z["node_ids"]),
        )


@dataclass(frozen=True)
class Solution:
    routes: tuple[tuple[int, ...], ...]
    solver: str = ""
    runtime_s: float = 0.0

    @classmethod
    def from_lists(
        cls, routes: list[list[int]], solver: str = "", runtime_s: float = 0.0
    ) -> Solution:
        return cls(tuple(tuple(int(v) for v in r) for r in routes if r), solver, runtime_s)

    @property
    def num_routes(self) -> int:
        return len(self.routes)

    def giant_tour(self) -> list[int]:
        """Routes joined with depot separators: [0, a, b, 0, c, 0]."""
        tour = [0]
        for r in self.routes:
            tour.extend(r)
            tour.append(0)
        return tour
