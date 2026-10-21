"""Random instances for tests and sanity checks (the calibrated generator comes in phase 3)."""

from __future__ import annotations

import numpy as np

from neural_routing.core.instance import VRPInstance


def random_instance(
    num_customers: int,
    *,
    seed: int = 0,
    capacity: float = 50.0,
    num_vehicles: int = 1,
    with_time_windows: bool = False,
    speed: float = 1.0,
    service_time: float = 10.0,
) -> VRPInstance:
    """Uniform points in [0, 1000]^2, Euclidean travel times, integer demands in [1, 9]."""
    rng = np.random.default_rng(seed)
    n = num_customers + 1
    coords = rng.uniform(0, 1000, size=(n, 2))
    durations = np.linalg.norm(coords[:, None] - coords[None], axis=-1) / speed
    demands = rng.integers(1, 10, size=n).astype(np.float64)
    demands[0] = 0
    service = np.full(n, service_time)
    service[0] = 0
    tw = np.tile([0.0, np.inf], (n, 1))
    if with_time_windows:
        # windows centred on the direct-trip arrival time so most instances stay feasible
        centre = durations[0] + rng.uniform(0, 4000, size=n)
        width = rng.uniform(1500, 4000, size=n)
        tw[1:, 0] = np.maximum(0, centre[1:] - width[1:] / 2)
        tw[1:, 1] = centre[1:] + width[1:] / 2
    return VRPInstance(
        name=f"rand{num_customers}_s{seed}",
        coords=coords,
        durations=durations,
        demands=demands,
        service_times=service,
        time_windows=tw,
        capacity=capacity,
        num_vehicles=num_vehicles,
    )
